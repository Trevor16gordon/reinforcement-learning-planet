""" 
Entry point for training the State Space Model


This model will
- Instantiate the VAE from models.py
- Instantiate the SSM from models.py (Stochastic or Deterministic or Recurrent)
- Use the data loader to get data
"""

from env import GymEnv, GYM_ENVS, CONTROL_SUITE_ENVS 
from data import ExperienceReplay
from Models import MODEL_DICT
from utils import gather_data, compute_loss

import torch
from torch import optim, nn

# from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import numpy as np
import argparse
import yaml
import os
import glob
import time
import cv2
import gym



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--save-path", type=str, default="", help="Path for saving model check points.")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument( "--env", type=str, default="MountainCar-v0", help="Gym/Control Suite environment")
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
    """
    if args.env in GYM_ENVS:
        env = GymEnv 
    else:
        # create comparable wrapper for control suite tasks
        raise NotImplementedError("No Control Suite Wrapper written yet.")
    """
    env = GymEnv
    env = env(
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

    gather_data(env, memory, config["seed_episodes"])

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
        'test_rewards': []
    }

    for itr in range(train_config["train_iters"]):
        kl_loss, obs_loss, rew_loss, loss = compute_loss(
            transition_model,
            memory,
            kl_clip,
            global_prior_means,
            global_prior_stddvs,
            train_config,
            args.model
        )    

        losses["kl_loss"].append(kl_loss.item())
        losses["obs_loss"].append(obs_loss.item())
        losses["rew_loss"].append(rew_loss.item())
        losses["sum_loss"].append(loss.item())

        print(loss.item())

        # standard back prop step. Includes gradient clipping to help with training the RNN.
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(transition_model.parameters(), train_config["clip_grad_norm"], norm_type=2)
        optimiser.step()

        if ((itr + 1) % config["checkpoint_interval"]) == 0:
            model_save_info = {
                "state_dict" : transition_model.state_dict(),
                "env_name": args.env,
                "model_config": model_config,
                "model": args.model,
                "env_config": config["env"],
                "seed": args.seed,
            }
            torch.save(model_save_info, os.path.join(args.save_path, f"{args.model}_{itr}.pkl"))
