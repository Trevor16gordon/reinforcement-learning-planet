""" 
Entry point for training the State Space Model


This model will
- Instantiate the VAE from models.py
- Instantiate the SSM from models.py (Stochastic or Deterministic or Recurrent)
- Use the data loader to get data
"""

from env import GymEnv, ControlSuiteEnv
from data import ExperienceReplay
from Models import SSM, MODEL_DICT
from utils import gather_data, compute_loss, rollout_using_mpc
from dynamics import ModelPredictiveControl, LearnedDynamics

import torch
from torch import optim, nn
from torch.distributions import Normal

# from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from argparse import Namespace
import numpy as np
import argparse
import yaml
import os
import glob
import time
import cv2
import gym
import pdb
import copy



GYM_ENVS = ["InvertedPendulum-v2", "Pendulum-v1", "MountainCar-v0", "CartPole-v1"]
CONTROL_SUITE_ENVS = ["ant-v2", "cartpole-swingup"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument( "--env", type=str, default="CartPole-v1", help="Gym/Control Suite environment")
    parser.add_argument( "--model", type=str, default="ssm", choices=["ssm", "rssm", "rnn"], help="Select the State Space Model to Train.",)
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    parser.add_argument( "--config", type=str, default="base", help="Specify the yaml file to use in setting up experiment.",)
    parser.add_argument( "--config-path", type=str, default="Configs", help="Specify the directory the config file lives in.")

    args = parser.parse_args()

    # load in congig file and update with the command line arguments.
    with open(f"{args.config_path}/{args.config}.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"\tExperiment ID: {args.id} \n\tRunning on device: {device}")


    # set up directory for writing experiment results to.
    results_dir = os.path.join("results", args.id)
    os.makedirs(results_dir, exist_ok=True)

    # Set the initial keys for numpy, torch, and the GPU.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # instantiate the environment and the Experience replay buffer to collect trajectories.
    
    if args.env in GYM_ENVS:
        envClass = GymEnv
    elif args.env in CONTROL_SUITE_ENVS:
        envClass = ControlSuiteEnv
    else:
        # create comparable wrapper for control suite tasks
        raise NotImplementedError("No Control Suite Wrapper written yet.")
    
    env = envClass(
        args.env,
        args.seed,
        **config["env"],
    )
    memory = ExperienceReplay(
        env.action_size,
        env.observation_size,
        device,
        config["memory"]["bit_depth"],
        config["memory"]["size"],
        config["memory"]["symbolic_env"],
    )

    # Select the model to instantiate, and then access its hyperparameters in the config file.
    model_config = config[args.model]
    transition_model = MODEL_DICT[args.model](
        env.observation_size, env.action_size, device, **model_config
    ).to(device)
    transition_model.train(True)

    max_train_episode_reward = gather_data(env, memory, config["seed_episodes"])

    train_config = config["train"]
    optimiser = optim.Adam(
        transition_model.parameters(),
        lr=train_config["learning_rate"],
        eps=train_config["adam_epsilon"],
    )

    # The global prior is used to compute an additional KL-divergence loss term between the posterior values
    # and the the unit normal distribtuion. This acts in the same way to the KL loss term in a standard VAE.
    global_prior_means = torch.zeros(
        train_config["batch_size"],
        model_config["state_size"],
        dtype=torch.float32,
        device=device,
    )
    global_prior_stddvs = torch.ones(
        train_config["batch_size"],
        model_config["state_size"],
        dtype=torch.float32,
        device=device,
    )

    # Upper bound on the KL-Loss term. Added to help stabalize training. Can experiment with changing the scale of this term.
    kl_clip = torch.full(
        (1,),
        train_config["kl_clip"],
        dtype=torch.float32, 
        device=device
    )
    
    # Track the loss for each of the individual model loss components.
    losses = {
        "obs_loss": [],
        "rew_loss": [],
        "kl_loss":  [],
        "sum_loss": [],
        "LOS_loss": [],
        "GP_loss":  []
    }
    # misc. metrics of interest for later plotting and visualization.
    metrics = {
        'steps': [],
        'episodes': [], 
        'train_rewards': [], 
        'test_episodes': [], 
        'test_rewards': [],
        'max_train_episode_reward': []
    }
    total_test_reward = 0
    num_test = 1
    
    for iter in range(train_config["train_iters"]):
        loss = 0

        # Sample batch_size sequences of length at random from the replay buffer.
        # Our sampled transitions are treated as starting from a random intial state at time 0.
        observations, actions, rewards, nonterminals = memory.sample(
            train_config["batch_size"], train_config["seq_length"]
        )
        
        # we start all models with an initial state and belief of 0's
        init_belief = torch.zeros( train_config["batch_size"], model_config["belief_size"]).to(device)
        init_state = torch.zeros( train_config["batch_size"], model_config["state_size"]).to(device)

        # encode the observations by passing them through the models encoder network.
        encoded_observations = transition_model.encode(observations)
        (
            beliefs,
            prior_states,
            prior_means,
            prior_stddvs,
            posterior_states,
            posterior_means,
            posterior_stddvs,
            reward_preds,
        ) = transition_model(
            init_state,
            actions[:-1],
            init_belief,
            encoded_observations[1:],
            nonterminals[:-1]
        )


        # The loss function is the sum of the reconstruction loss, the reward prediction loss, and the KL-divergence loss.
        kl_loss = transition_model.kl_loss(prior_means, prior_stddvs, posterior_means, posterior_stddvs, kl_clip)
        obs_loss = transition_model.observation_loss(posterior_states, beliefs, observations[1:])
        rew_loss = transition_model.reward_loss(reward_preds, rewards[:-1])
        loss = kl_loss + obs_loss + rew_loss

        losses["kl_loss"] = kl_loss.item()
        losses["obs_loss"] = obs_loss.item()
        losses["rew_loss"] = rew_loss.item()
        losses["sum_loss"] = loss.item()

        if 0 < train_config["global_kl_beta"]:
            loss += train_config["global_kl_beta"] * transition_model.kl_loss(
                global_prior_means,
                global_prior_stddvs,
                posterior_means,
                posterior_stddvs,
                kl_clip,
            )

        # standard back prop step. Includes gradient clipping to help with training the RNN.
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(transition_model.parameters(), train_config["grad_clip_norm"], norm_type=2)
        optimiser.step()

        print(losses)

        # Data Collection using MPC
        transition_model.eval()
        dyn = LearnedDynamics(args.env, transition_model, env.action_size, env.observation_size)
        rollout_using_mpc(
            dyn,
            transition_model,
            env,
            config["mpc_data_collection"],
            config["max_episode_len"],
            memory=memory,
            action_noise_variance=0.03)
        transition_model.train(True)

        print(losses)

        if ((iter + 1) % config["checkpoint_interval"]) == 0:
            model_save_info = {
                "state_dict" : transition_model.state_dict(),
                "env_name": args.env,
                "model_config": model_config,
                "model": args.model,
                "env_config": config["env"],
                "seed": args.seed,
                }
            torch.save(model_save_info, f"transition_model2_{iter}.pkl")

        if ((iter + 1) % config["test_interval"]) == 0:
            # Test performance using MPC
            transition_model.eval()
            dyn = LearnedDynamics(args.env, transition_model, env.action_size, env.observation_size)
            avg_reward_per_episode = rollout_using_mpc(
                dyn,
                transition_model,
                env,
                config["mpc"],
                config["max_episode_len"],
                memory=None,
                action_noise_variance=None)
            total_test_reward += avg_reward_per_episode
            print(f"Test episode completed. Survived {i} episodes. Average tst reward so far {total_test_reward/num_test} Last test reward {avg_reward_per_episode}")
            num_test += 1
            transition_model.train(True)

