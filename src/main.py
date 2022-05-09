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
from utils import gather_data, compute_loss, rollout_using_mpc, update_belief_and_act, write_video
from dynamics import ModelPredictiveControl, LearnedDynamics
from env import (
        GymEnv,
        ControlSuiteEnv, 
        GYM_ENVS,
        CONTROL_SUITE_ENVS, 
        CONTROL_SUITE_ACTION_REPEATS 
    )

import torch
from torch import optim, nn

import numpy as np
import argparse
import yaml
import os
import tqdm
import time
import tqdm

GYM_ENVS = ["InvertedPendulum-v2", "Pendulum-v1", "MountainCar-v0", "CartPole-v1"]
CONTROL_SUITE_ENVS = ["ant-v2", "cartpole-swingup"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--save-path", type=str, default="", help="Path for saving model check points.")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument( "--env", type=str, default="MountainCar-v0", choices=GYM_ENVS+CONTROL_SUITE_ENVS, help="Gym/Control Suite environment")
    parser.add_argument( "--model", type=str, default="ssm", choices=list(MODEL_DICT.keys()), help="Select the State Space Model to Train.",)
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    parser.add_argument( "--config", type=str, default="base", help="Specify the yaml file to use in setting up experiment.",)
    parser.add_argument( "--config-path", type=str, default="Configs", help="Specify the directory the config file lives in.")

    args = parser.parse_args()

    # load in congig file and update with the command line arguments.
    with open(f"{args.config_path}/{args.config}.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"\tExperiment ID: {args.id} \n\tRunning on device: {device}\n")


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
        config["env"]["action_repeat"] = CONTROL_SUITE_ACTION_REPEATS[args.env.split('-')[0]] 
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
    transition_model.train()


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

    global_start_time = time.time()

    # populate the memory buffer with random action data.
    gather_data(env, memory, train_config["seed_episodes"])

    # Collect N episodes. Train the model at the end of each new episode. Intermitently run
    # 10 episodes of MPC to evaluate the models current performanc. 
    for traj in range(train_config["episodes"]):

        # Training
        for itr in tqdm.tqdm(range(train_config["train_iters"])):
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

            # standard back prop step. Includes gradient clipping to help with training the RNN.
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(transition_model.parameters(), train_config["clip_grad_norm"], norm_type=2)
            optimiser.step()

        # enerate a video of the original trajectory alongside the models reconstruction.
        if traj % 25 == 0:
            transition_model.eval()

            with torch.no_grad():
                observation, total_reward, video_frames = env.reset(), 0, []
                belief = torch.zeros(1, model_config["belief_size"], device=device)
                posterior_state = torch.zeros(1, model_config["state_size"], device=device)
                action = -1 + 2*torch.rand(1, env.action_size, device=device)

                for _ in range(500):
                    belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                        env, 
                        transition_model, 
                        belief,
                        posterior_state,
                        action, 
                        observation.to(device=device),
                    )
                    video_frames.append(
                        (torch.cat([observation.squeeze(), transition_model.decode(belief, posterior_state).squeeze().cpu()], dim=2) + 0.5).numpy()
                    )
                    observation = next_observation
                    if done:
                        env.close()
                        break
            write_video(video_frames, f"{args.model}_{args.env}_{traj}_episodes", "videos")
            transition_model.train()

        # Print Info
        if traj % 1 == 0:
            total_time = int(time.time() - global_start_time)
            total_secs, total_mins, total_hrs = total_time % 60, (total_time // 60) % 60, total_time // 3600
            print(f"Total Run Time: {total_hrs:02}:{total_mins:02}:{total_secs:02}\n" \
                  f"Trajectory {traj}: \n\tTotal Loss: {loss.item():.2f}" \
                  f"\n\tObservation Loss {obs_loss.item():.2f}"
                  f"\n\tReward Loss {rew_loss.item():.2f}"
                  f"\n\tKL Loss {kl_loss.item():.2f}\n"
            )

        if config["mpc_data_collection"]["optimization_iters"] == 0:
            # naive data collection for now. Eventually integrate the MPC code to collect data
            gather_data(env, memory, 1)

        else:
            # Data Collection using MPC
            transition_model.eval()
            dyn = LearnedDynamics(args.env, transition_model, env.action_size, env.observation_size)
            rollout_using_mpc(
                dyn,
                transition_model,
                env,
                config["mpc_data_collection"],
                memory=memory,
                action_noise_variance=config["mpc_data_collection"]["exploration_noise"])
            transition_model.train()

        if ((traj + 1) % config["test_interval"]) == 0:
            # Test performance using MPC
            transition_model.eval()
            dyn = LearnedDynamics(args.env, transition_model, env.action_size, env.observation_size)
            avg_reward_per_episode = rollout_using_mpc(
                dyn,
                transition_model,
                env,
                config["mpc"],
                memory=None,
                action_noise_variance=None
            )
            total_test_reward += avg_reward_per_episode
            print(f"Test episode completed. Average test reward so far {total_test_reward/num_test} Last test reward {avg_reward_per_episode}")
            num_test += 1
            transition_model.train()
            
        if (traj + 1) % config["checkpoint_interval"] == 0:
            model_save_info = {
                "model_state_dict" : transition_model.state_dict(),
                "env_name": args.env,
                "model_config": model_config,
                "model": args.model,
                "env_config": config["env"],
                "seed": args.seed,
                }
            torch.save(model_save_info, f"transition_model_{traj}.pt")
