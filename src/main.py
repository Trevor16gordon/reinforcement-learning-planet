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
from utils import gather_data, compute_loss, rollout_using_mpc, write_video
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

import pandas as pd
import argparse
import yaml
import os
import tqdm
import time
import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--save-path", type=str, default="", help="Path for saving model check points.")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument( "--load-model-path", type=str, default=None, help="If given, model will start from this path")
    parser.add_argument( "--save-results-path", type=str, default="results", help="Path to output file")
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
    video_dir = os.path.join(results_dir, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Set the initial keys for numpy, torch, and the GPU.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # instantiate the environment and the Experience replay buffer to collect trajectories.
    if args.env in GYM_ENVS:
        envClass = GymEnv
    elif args.env in CONTROL_SUITE_ENVS:
        envClass = ControlSuiteEnv
        config["env"]["action_repeat"] = CONTROL_SUITE_ACTION_REPEATS[args.env.split("-")[0]] 
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
    optimizer = optim.Adam(
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
    }
    if train_config["overshooting_kl_beta"]:
        losses["overshooting_kl_loss"] = []
    if train_config["overshooting_reward_beta"]:
        losses["overshooting_reward_loss"] = []

    # misc. metrics of interest for later plotting and visualization.
    metrics = {
        "steps": [],
        "episodes": [], 
        "train_rewards": [], 
        "test_episodes": [], 
        "test_rewards": [],
        "test_reward_avg": []
    }
    total_test_reward = 0
    num_test = 1

    global_start_time = time.time()

    # populate the memory buffer with random action data.
    gather_data(env, memory, train_config["seed_episodes"])
    
    # dynamics wrapper for mpc.
    dyn = LearnedDynamics(args.env, transition_model, env.action_size, env.observation_size)

    # Collect N episodes. Train the model at the end of each new episode. Intermitently run
    # 10 episodes of MPC to evaluate the models current performanc. 
    for traj in range(train_config["episodes"]):

        # Set models to train mode
        transition_model.train()
        for itr in tqdm.tqdm(range(train_config["train_iters"])):
            kl_loss, obs_loss, rew_loss, overshooting_kl_loss, overshooting_reward_loss, loss = compute_loss(
                transition_model,
                memory,
                kl_clip,
                global_prior_means,
                global_prior_stddvs,
                train_config,
                args.model
            )    
            # standard back prop step. Includes gradient clipping to help with training the RNN.
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(transition_model.parameters(), train_config["clip_grad_norm"], norm_type=2)
            optimizer.step()

            losses["kl_loss"].append(kl_loss.item())
            losses["obs_loss"].append(obs_loss.item())
            losses["rew_loss"].append(rew_loss.item())
            losses["sum_loss"].append(loss.item())

            if overshooting_kl_loss is not None:
                losses["overshooting_kl_loss"].append(overshooting_kl_loss.item())
            if overshooting_kl_loss is not None:
                losses["overshooting_reward_loss"].append(overshooting_reward_loss.item())


        # generate a video of the original trajectory alongside the models reconstruction.
        if traj % 5 == 0:
            transition_model.eval()
            with torch.no_grad():
                _, video_frames = rollout_using_mpc(
                    dyn,
                    transition_model,
                    env,
                    config["mpc"],
                    memory=None,
                    action_noise_variance=None,
                    decode_to_video=True,
                    max_frames=config["save_video_n_frames"]
                )
            write_video(video_frames, f"{args.model}_{args.env}_{traj}_episodes", video_dir)
            transition_model.train()


        # Print Info
        total_time = int(time.time() - global_start_time)
        total_secs, total_mins, total_hrs = total_time % 60, (total_time // 60) % 60, total_time // 3600
        print(f"Total Run Time: {total_hrs:02}:{total_mins:02}:{total_secs:02}\n" \
              f"Trajectory {traj}: \n\tTotal Loss: {loss.item():.2f}" \
              f"\n\tObservation Loss: {obs_loss.item():.2f}"
              f"\n\tReward Loss: {rew_loss.item():.2f}"
              f"\n\tKL Loss: {kl_loss.item():.2f}"
        )
        if overshooting_kl_loss is not None:
            print(f"\tOS KL Loss: {overshooting_kl_loss.item():.2f}")
        if overshooting_reward_loss is not None:
            print(f"\tOS Reward Loss: {overshooting_reward_loss.item():.2f}")
        print()


        # Data Collection using MPC
        if config["mpc_data_collection"]["optimization_iters"] == 0:
            gather_data(env, memory, 1)
        else:
            with torch.no_grad():
                train_reward, _ = rollout_using_mpc(
                    dyn,
                    transition_model,
                    env,
                    config["mpc_data_collection"],
                    memory=memory,
                    action_noise_variance=config["mpc_data_collection"]["exploration_noise"],
                    decode_to_video=False,
                )
                        

        # Test Model with N trajectories.
        if (traj + 1) % config["test_interval"] == 0 or traj == 0:
            transition_model.eval()
            # Test performance using MPC
            test_episode_rewards = []
            with torch.no_grad():
                for _ in range(config["test_episodes"]):
                    test_episode_reward, vid_frames = rollout_using_mpc(
                        dyn,
                        transition_model,
                        env,
                        config["mpc"],
                        memory=None,
                        action_noise_variance=None,
                        decode_to_video=False,
                    )
                    test_episode_rewards.append(test_episode_reward)

            test_reward_avg = sum(test_episode_rewards)/len(test_episode_rewards)
            print(f"Test episode completed. Average test reward: {test_reward_avg}")
            num_test += 1
            transition_model.train()
            

        metrics["steps"].append(train_config["train_iters"]*traj)
        metrics["episodes"].append(traj)
        metrics["train_rewards"].append(train_reward)
        metrics["test_episodes"].append(num_test)
        metrics["test_rewards"].append(test_episode_rewards)
        metrics["test_reward_avg"].append(test_reward_avg)

        pd.DataFrame(losses).to_csv(os.path.join(results_dir, f"{args.model}_config_{args.config}_{args.id}_losses.csv"))
        pd.DataFrame(metrics).to_csv(os.path.join(results_dir, f"{args.model}_config_{args.config}_{args.id}_metrics.csv"))


        if (traj + 1) % config["checkpoint_interval"] == 0 or traj == 0:
            model_save_info = {
                "model_state_dict" : transition_model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "env_name": args.env,
                "model_config": model_config,
                "model": args.model,
                "env_config": config["env"],
                "seed": args.seed,
            }
            torch.save(model_save_info, os.path.join(results_dir, f"{args.model}_config_{args.config}_{args.id}_{traj}.pt"))
