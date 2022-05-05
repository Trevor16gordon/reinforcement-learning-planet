""" 

This file is intended to be a catch all for misc functions

"""

from typing import Optional, List

from data import ExperienceReplay
from Models.base import TransitionModel
from env import BaseEnv

import numpy
import torch


def gather_data(
    env: BaseEnv, 
    memory: ExperienceReplay, 
    n_trajectories: int = 5
) -> None:
    """
    Gather N trajectories using random actions and add the transitions to the
    experience replay memory.
    """

    for _ in range(n_trajectories):
        state = env.reset()
        done = False
        while not done:
            action = env.sample_random_action()
            next_state, reward, done, info = env.step(action)
            memory.append(next_state, action, reward, done)

    env.close()


def compute_loss(
    transition_model: TransitionModel,
    memory: ExperienceReplay,
    kl_clip: torch.Tensor,
    global_prior_means: torch.Tensor,
    global_prior_stddvs: torch.Tensor,
    train_config: dict
) -> List[torch.Tensor]:
    pass
