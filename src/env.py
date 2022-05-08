import logging
from abc import ABC, abstractmethod

import gym
from gym import spaces
from gym.wrappers import TimeLimit
from gym.wrappers.pixel_observation import PixelObservationWrapper

import numpy as np
import torch
import cv2

from data import images_to_observation

GYM_ENVS = [
    'Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2',
    'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 
    'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2'
]
CONTROL_SUITE_ENVS = [
    'cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 
    'cheetah-run', 'ball_in_cup-catch', 'walker-walk'
]
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}

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
    def reset(self):
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

        self.symbolic = symbolic_env
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        
        # set up the gym env and wrap with PixelObservationWrapper
        # to access pixel level observations of the env.
        self._env = gym.make(env)
        self._env.seed(seed)
        self._env.reset()
        self._env = TimeLimit(self._env, max_episode_length)
        self._env = PixelObservationWrapper(self._env)

        # update the observations space to account for the new [3, 64, 64]
        # dimensions, as opposed to base env observations space.
        self._env.observation_space = spaces.Box(
            shape=[3, 64, 64], low=0, high=255, dtype=np.float
        )


    def step(self, action):
        # If we are repeating actions in the EVN, then the return reward
        # is the sum of the recieved rewards over the time frame
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            
            if done:
                break

        # if the env returns both pixels and extra information in the state
        # we want to extract the pixel data from the returned dict.
        if isinstance(state, dict):
            state = state["pixels"]
        observation = images_to_observation(state, self.bit_depth)
        return observation, reward, done, info

    # pass standard method calls down to the underlying gym environment.
    def reset(self):
        # Reset internal timer
        state = self._env.reset()

        # if the env returns both pixels and extra information in the state
        # we want to extract the pixel data from the returned dict.
        if isinstance(state, dict):
            state = state["pixels"]
        
        observation = images_to_observation(state, self.bit_depth)
        return observation

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
        act_shape = self._env.action_space.shape
        if 1 <= len(act_shape):
            return act_shape[0]
        return 1

    @property
    def action_range(self):
        return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return np.array(self._env.action_space.sample())



class ControlSuiteEnv():
    """Borrowed from https://github.com/Kaixhin/PlaNet/blob/master/env.py"""
    def __init__(
        self,
        env: str,
        seed: int, 
        symbolic_env: bool,
        max_episode_length: int, 
        action_repeat: int, 
        bit_depth: int
        ):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic_env
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic_env:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
        else:
            return images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break

            observation = images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        info = {}
        return observation, reward, done, info

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    @property
    def action_range(self):
        return float(self._env.action_spec().minimum[0]), float(self._env.action_spec().maximum[0]) 

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))
