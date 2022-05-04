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


def gather_data(env, memory: ExperienceReplay, n_trajectories: int = 5) -> None:
    """
        Gather N trajectories using random actions and add the transitions to the 
        experience replay memory.
    """
    
    for _ in range(n_trajectories):
        state = env.reset()
        done = False
        while not done: 
            action = np.stack([env.action_space.sample() for _ in range(num_envs)]) # Shape is (num_envs, action_dim)
            next_state, reward, done, info  = env.step(action)
            obsv_i = env.get_images()[0]
            obsv_i = cv2.resize(obsv_i,  dsize=(64, 64))
            obsv_i = np.swapaxes(obsv_i, 0, 2)
            memory.append(obsv_i, action[0, 0], reward[0], False)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="base", help="Specify the yaml file to use in setting up experiment.")
    parser.add_argument("--config-path", type=str, default="Configs", help="Specify the directory the config file lives in.")
    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument("--env", type=str, default="InvertedPendulum-v2", help="Gym/Control Suite environment")
    parser.add_argument("--model", type=str, default="ssm", choices=["ssm", "rssm", "rnn"], help="Select the State Space Model to Train.")
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    
    args = parser.parse_args()
    
    # load in congig file and update with the command line arguments.
    with open(f"{args.config_path}/{args.config}.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    config.update(vars(args))
    
    # set up directory for writing experiment results to.
    results_dir = os.path.join("results", config["id"])
    os.makedirs(results_dir, exist_ok=True)

    # Set the initial keys for numpy, torch, and the GPU.
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    config["device"] = torch.device("cuda")
    torch.cuda.manual_seed(config["seed"])
    
    # instantiate the environment and the Experience replay buffer to collect trajectories.
    env = DummyVecEnv([lambda : gym.make(config["env"]) for i in range(1)])
    replay_memory = ExperienceReplay(
        config["memory"]["experience_size"],
        config["symbolic_env"],
        env.observation_size, 
        env.action_size, 
        config["memory"]["bit_depth"], 
        config["device"]
    )

    # Select the model to instantiate, and then access its hyperparameters in the config file.
    model_config = config[args.model]
    model_config["obs_dim"] = env.observation_size, 
    model_config["act_size"] = env.action_size, 
    transition_model = MODEL_DICT[args.model](**model_config).to(config["device"])


    

