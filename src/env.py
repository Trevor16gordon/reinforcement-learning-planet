import logging
from abc import ABC, abstractmethod

import gym
from gym import spaces
from gym.wrappers.pixel_observation import PixelObservationWrapper

import numpy as np
import torch
import cv2

from data import images_to_observation


class BaseEnv(ABC):
    """
        Base Class to ensure that all our environments share the same necessary methods and properties. 
    """

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def sample_random_action(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_range(self):
        raise NotImplementedError


class GymEnv(BaseEnv):
    """
        Wrap standard gym environments with the pixel observation wrapper. 
        Preprocess the returned images using the images_to_obervation
        function to place the data in the correct domain ([-0.5, 0.5])
    """

    def __init__(
        self,
        env: str,
        seed: int, 
        symbolic_env: bool,
        max_episode_length: int, 
        action_repeat: int, 
        bit_depth: int
    ):  
        # Ignore warnings from Gym logger
        gym.logger.set_level(logging.ERROR)

        # set up the gym env and wrap with PixelObservationWrapper
        # to access pixel level observations of the env.
        self._env = gym.make(env)
        self._env.seed(seed)
        self._env = PixelObservationWrapper(self._env, render_kwargs={'mode':'rgb_array'})

        self.symbolic = symbolic_env
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        
        # update the observations space to account for the new [3, 64, 64]
        # dimensions, as opposed to base env observations space.
        self._env.observation_space = spaces.Box(
            shape=[3, 64, 64], low=0, high=255, dtype=np.float
        )


    def step(self, action):
        action = action.detach().numpy()

        # If we are repeating actions in the EVN, then the return reward
        # is the sum of the recieved rewards over the time frame
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  
            
            if done or self.t == self.max_episode_length:
                break

        observation = _images_to_observation(state["pixels"], self.bit_depth)
        return observation, reward, done

    # pass calls the underlying gym environment.
    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        """ our environments will always return (3, 64, 64) images."""
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return np.array(self._env.action_space.sample())
