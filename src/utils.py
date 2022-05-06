""" 

This file is intended to be a catch all for misc functions

"""

from typing import Optional, List

from data import ExperienceReplay
from Models.base import TransitionModel
from env import BaseEnv, GymEnv
from Models import SSM, MODEL_DICT

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

def gather_data_for_testing(
    env_name: str = "CartPole-v1",
    rollout_len: int = 10
) -> None:
    """
    Gather data for one trajectory and return a sequence of rollout_len
    """
    env_wrapped = GymEnv(env_name, 1, False, 10000, 1, 8)
    action_size = env_wrapped.action_size
    observation_size = env_wrapped.observation_size
    dev = torch.device("cpu")
    memory = ExperienceReplay(action_size, observation_size, dev, 8, 1000, False)
    gather_data(env_wrapped, memory, 1)
    observations, actions, rewards, nonterminals = memory.sample(1, rollout_len)
    return (observations, actions, rewards, nonterminals)

def gather_reconstructed_images_from_saved_model(path_to_model, rollout_len=10):
    """Load a saved model and generated original and reconstructed VAE images

    Args:
        path_to_model (str): Path to .pkl model file
            The saved checkpoint must contain the following keys:
            state_dict, env_name, model_config, model, env_config, seed
        rollout_len (int, optional): Number of images to roll out in a single episode. Defaults to 10.

    Returns:
         original_images, reconstructed_images (np.array, np.array):
    """

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(path_to_model)

    assert "state_dict" in checkpoint.keys()
    assert "env_name" in checkpoint.keys()
    assert "model_config" in checkpoint.keys()
    assert "model" in checkpoint.keys()
    assert "env_config" in checkpoint.keys()
    assert "seed" in checkpoint.keys()

    env_name = checkpoint["env_name"]
    model = checkpoint["model"]
    env = GymEnv(
        checkpoint["env_name"],
        checkpoint["seed"],
        **checkpoint["env_config"],
    )
    
    transition_model = MODEL_DICT[model](
        env.observation_size, env.action_size, device, **checkpoint["model_config"]
    )

    transition_model.load_state_dict(checkpoint["state_dict"])
    (observations, actions, rewards, nonterminals) = gather_data_for_testing(env_name, rollout_len)

    init_belief = torch.zeros(1, 0)
    init_state = torch.zeros(1, 30)

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

    reconstructed_obs = transition_model._decoder(posterior_states, beliefs)
    reconstructed_images = reconstructed_obs.permute(0, 2, 3, 1).detach().numpy()
    original_images = observations.squeeze().permute(0, 2, 3, 1).detach().numpy()

    return original_images, reconstructed_images
    

def compute_loss(
    transition_model: TransitionModel,
    memory: ExperienceReplay,
    kl_clip: torch.Tensor,
    global_prior_means: torch.Tensor,
    global_prior_stddvs: torch.Tensor,
    train_config: dict
) -> List[torch.Tensor]:
    pass
