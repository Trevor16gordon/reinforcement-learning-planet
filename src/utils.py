""" 

This file is intended to be a catch all for misc functions

"""

from typing import Optional, List

from data import ExperienceReplay
from Models.base import TransitionModel
from env import BaseEnv, GymEnv
from Models import SSM, MODEL_DICT
from dynamics import ModelPredictiveControl

import copy
import numpy
import torch
import numpy as np
import pdb

def gather_data(
    env: BaseEnv, 
    memory: ExperienceReplay, 
    n_trajectories: int = 5,
) -> None:
    """
    Gather N trajectories using random actions and add the transitions to the
    experience replay memory.
    """
    max_episode_reward_tot = 0
    for _ in range(n_trajectories):
        state = env.reset()
        done = False
        i = 0
        episode_reward_tot = 0
        while not done:
            action = env.sample_random_action()
            next_state, reward, done, info = env.step(action)
            memory.append(next_state, action, reward, done)
            i += 1
            episode_reward_tot += reward
        max_episode_reward_tot = max(max_episode_reward_tot, episode_reward_tot)
    env.close()
    return max_episode_reward_tot

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

def load_model_from_path(path_to_model):
    """Load a saved model

    Args:
        path_to_model (str): Path to .pkl model file
            The saved checkpoint must contain the following keys:
            state_dict, env_name, model_config, model, env_config, seed
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
    return transition_model, checkpoint

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

    transition_model, checkpoint = load_model_from_path(path_to_model)
    env_name = checkpoint["env_name"]
    
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

def rollout_using_mpc(dyn, transition_model_mpc, env, mpc_config, memory=None, action_noise_variance=None):
    """Rollout an episode using the MPC to choose the best actions

    Args:
        dyn (LearnedDynamics): 
        transition_model_mpc (Models.Base): The trained transition model
        env (BaseEnv): The environment. It will be reset in this function
        mpc_config (dict): See mpc config in Configs.base.yaml
        max_episode_len (int): The max length of episode
        memory (ExperienceReplay, optional): If given, the experience will be added to the memory buffer
        action_noise_variance (int, optional): If given, uniform noise with this variance will be added to the action
    """
    mpc = ModelPredictiveControl(
            dyn, 
            min_action_clip=env.action_range[0],
            max_action_clip=env.action_range[1])
    mpc.control_horizon_simulate = mpc_config["planning_horizon"]
    state = env.reset().squeeze()
    (generated_t0_rewards,
    generated_t0_prior_states,
    generated_t0_beliefs) = transition_model_mpc.forward_generate(torch.zeros(1, 1, 1), obs_0=state)

    current_state = generated_t0_prior_states[0]
    current_belief = generated_t0_beliefs
    # Need to repeat the prev_state and prev_belief batch number of times
    current_state_repeat = current_state.repeat(mpc_config["candidates"], 1)
    dyn.model_belief = current_belief
    dyn.model_state = current_state_repeat
    # Calculate belief_0 and prev_state_0: Might need to reshape as batch dimension will be 1
    avg_reward_per_episode = 0
    done = False
    while not done:
        best_actions = mpc.compute_action_cross_entropy_method(
            state, 
            None, # No goal as using sum of rewards to select best action sequence
            num_iterations=mpc_config["optimization_iters"],
            j=mpc_config["candidates"], 
            k=mpc_config["top_candidates"])
        action = best_actions[0, :]

        if action_noise_variance is not None:
            action += np.random.normal(loc=0, scale=action_noise_variance, size=action.shape)
        next_state, reward, done, info  = env.step(action)
        if memory is not None:
            memory.append(next_state, action, reward, done)
        # Adding MPC test data to memory buffer as well
        # memory.append(next_state, action, reward, done)
        # Update for transition model keeping track of chosen states
        action_torch = torch.ones(1, 1, env.action_size)
        action_torch[0, 0, :] = torch.from_numpy(action)
        (generated_rewards,
        generated_prior_states,
        generated_beliefs) = transition_model_mpc.forward_generate(action_torch, prev_state=current_state, prev_belief=current_belief)
        # Update current_state and current_belief
        current_state = generated_prior_states[0]
        current_state_repeat = current_state.repeat(mpc_config["candidates"], 1)
        current_belief = generated_t0_beliefs
        dyn.model_belief = current_belief
        dyn.model_state = current_state_repeat 
        avg_reward_per_episode += reward
        state = next_state.squeeze()
    print(f"avg_reward_per_episode is {avg_reward_per_episode}")
    return avg_reward_per_episode

def compute_loss(
    transition_model: TransitionModel,
    memory: ExperienceReplay,
    kl_clip: torch.Tensor,
    global_prior_means: torch.Tensor,
    global_prior_stddvs: torch.Tensor,
    train_config: dict
) -> List[torch.Tensor]:
    pass
