""" 
Entry point for training the State Space Model


This model will
- Instantiate the VAE from models.py
- Instantiate the SSM from models.py (Stochastic or Deterministic or Recurrent)
- Use the data loader to get data
    - Data should be compressed using the VAE before training
"""
from stable_baselines3.common.vec_env import DummyVecEnv
from config import GlobalConfig, SSMConfig, TrainingConfig
from pathlib import Path
from data import ExperienceReplay
import numpy as np
import argparse
import os
import glob
import time
import cv2
import gym

def train(args):


    D = ExperienceReplay(args.experience_replay_size, False, 0, 1, 8, "cpu")


    env_name = "InvertedPendulum-v2"
    num_envs = 1
    env = DummyVecEnv([lambda : gym.make(env_name) for i in range(num_envs)])


    state = env.reset()
    for i in range(args.batch_size):
        action = np.stack([env.action_space.sample() for _ in range(num_envs)]) # Shape is (num_envs, action_dim)
        next_state, reward, done, info  = env.step(action)
        obsv_i = env.get_images()[0]
        obsv_i = cv2.resize(obsv_i,  dsize=(64, 64))
        obsv_i = np.swapaxes(obsv_i, 0, 2)
        D.append(obsv_i, action[0, 0], reward[0], False)


    

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=SSMConfig.NUM_EPOCHS,
                        help="Number of epochs to train.", type=int)
    parser.add_argument("--learning_rate", required=False, default=SSMConfig.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", required=False, default=SSMConfig.HIDDEN_DIM,
                        help="The number of elements in hidden dimension")
    parser.add_argument("--model_dir", required=False,
                        default=GlobalConfig.get("MODEL_DIR"))
    parser.add_argument("--batch_size", required=False, type=int,
                        default=SSMConfig.BATCH_SIZE)
    parser.add_argument("--experience_replay_size", required=False, type=int,
                        default=TrainingConfig.EXPERIENCE_REPLAY_SIZE)
    parser.add_argument("--model_in_folder", required=False,
        help="Path to a folder containing disc / generator weights. The latest one will be loaded.")
    args = parser.parse_args()

    train(args)