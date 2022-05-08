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
    train_config: dict,
    model_type: str
) -> List[torch.Tensor]:
    """
    Shared Loss function for all models. All models have the following 3 methods:
        model.kl_loss
        model.obervation_loss
        model.reward_loss
    """
    loss = 0

    # Sample batch_size sequences of length at random from the replay buffer.
    # Our sampled transitions are treated as starting from a random intial state at time 0.
    observations, actions, rewards, nonterminals = memory.sample(
        train_config["batch_size"], train_config["seq_length"]
    )

    # we start all models with an initial state and belief of 0's
    init_belief = torch.zeros( train_config["batch_size"], model_config["belief_size"]).to(device)
    init_state = torch.zeros( train_config["batch_size"], model_config["state_size"]).to(device)

    # encode the observations by passing them through the models encoder network.
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

    # The loss function is the sum of the reconstruction loss, the reward prediction loss, and the KL-divergence loss.
    kl_loss = transition_model.kl_loss(prior_means, prior_stddvs, posterior_means, posterior_stddvs, kl_clip)
    obs_loss = transition_model.observation_loss(posterior_states, beliefs, observations[1:])
    rew_loss = transition_model.reward_loss(reward_preds, rewards[:-1])
    loss = kl_loss + obs_loss + rew_loss

    # The Global KL divergence term is not applicable to the RNN based Transition model.
    if 0 < train_config["global_kl_beta"] and model_type != 'rnn':
        loss += train_config["global_kl_beta"] * transition_model.kl_loss(
            global_prior_means,
            global_prior_stddvs,
            posterior_means,
            posterior_stddvs,
            kl_clip,
        )

    return kl_loss, obs_loss, rew_loss, loss 
