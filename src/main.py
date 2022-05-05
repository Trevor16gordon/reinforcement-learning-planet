""" 
Entry point for training the State Space Model


This model will
- Instantiate the VAE from models.py
- Instantiate the SSM from models.py (Stochastic or Deterministic or Recurrent)
- Use the data loader to get data
"""

from env import GymEnv
from data import ExperienceReplay
from Models import SSM
from utils import gather_data

import torch
from torch import optim, nn

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


MODEL_DICT = {
    "ssm": SSM,
}
GYM_ENVS = ["Pendulum-v1", "MountainCar-v0"] 
CONTROL_SUITE_ENVS = ["ant-v2"]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym/Control Suite environment")
    parser.add_argument("--model", type=str, default="ssm", choices=["ssm", "rssm", "rnn"], help="Select the State Space Model to Train.")
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    parser.add_argument("--config", type=str, default="base", help="Specify the yaml file to use in setting up experiment.")
    parser.add_argument("--config-path", type=str, default="Configs", help="Specify the directory the config file lives in.")
    
    args = parser.parse_args()
    
    # load in congig file and update with the command line arguments.
    with open(f"{args.config_path}/{args.config}.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    device = torch.device("cuda")
    
    # set up directory for writing experiment results to.
    results_dir = os.path.join("results", args.id)
    os.makedirs(results_dir, exist_ok=True)

    # Set the initial keys for numpy, torch, and the GPU.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # instantiate the environment and the Experience replay buffer to collect trajectories.
    if args.env in GYM_ENVS:
        env = GymEnv 
    else:
        # create comparable wrapper for control suite tasks
        raise NotImplementedError("No Control Suite Wrapper written yet.")

    env = env(
        args.env, 
        args.seed,
        **config["env"],
    )
    replay_memory = ExperienceReplay(
        env.observation_size, 
        env.action_size, 
        device,
        **config["memory"]
    )

    # Select the model to instantiate, and then access its hyperparameters in the config file.
    model_config = config[args.model]
    transition_model = MODEL_DICT[args.model](
        env.observation_size, 
        env.action_size, 
        **model_config
    ).to(device)
