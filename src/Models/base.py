"""

Implimentation of the Gaussian Stochastic State Space Model (SSM)

"""

from typing import Optional, List
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from Models.autoencoder import PixelObservationModel, PixelEncoder


class TransitionModel(ABC):
    """
    Base Class for all Transition Models.

    All inheriting classes should support the following methods:

        encode: pass pixel/symbolic level observations through the encoder.
        decode: pass the latent state and belief state through the decoder.
        forward: standard foward pass through the transition model.
        loss: Compute the model loss using the KL Divergence loss, the reconstruction loss, and the reward loss.
    """

    def __init__(
        self,
        obs_dim,
        act_size,
        device,
        state_size,
        embed_size,
        belief_size,
        hidden_dim,
        activation="ELU",
        min_stddev=1e-5,
    ):
        self._encoder = PixelEncoder(embed_size, obs_dim, activation="ReLU")
        self._decoder = PixelObservationModel(
            belief_size, state_size, embed_size, activation="ReLU"
        )
        self._obs_dim = obs_dim
        self._device = device

        self._state_size = state_size
        self._act_size = act_size
        self._embed_size = embed_size
        self._belief_size = belief_size
        self._hidden_dim = hidden_dim
        self._min_stddev = min_stddev
        
        self._activation = getattr(nn, activation)
        self.mse = nn.MSELoss(reduction="none")
        self.softplus = nn.Softplus()

    def kl_loss(
        self,
        prior_means: torch.Tensor,
        prior_stddev: torch.Tensor,
        posterior_means: torch.Tensor,
        posterior_stddev: torch.Tensor,
        max_divergence: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the KL-Divergence Term in the model loss function. 

        prior_means, prior_stddev, posterior_means, posterior_stddev should
        all be of the same dimension:
            [horizon, batch_size, state_dim]

        max_divergence is an upper bound on the KL_divergence loss, and is used to
        help stabalize training.
        """

        prior_dist = Normal(prior_means, prior_stddev)
        posterior_dist = Normal(posterior_means, posterior_stddev)
        kl_loss = kl_divergence(prior_dist, posterior_dist)

        if mask is not None:
            kl_loss = kl_loss * mask

        # sum accross the state dimension, and clip the loss.
        kl_loss = torch.max(kl_loss.sum(dim=2), max_divergence)

        # return the mean loss accross the time and batch dimensions.
        return kl_loss.mean(dim=(0, 1))

    def observation_loss(
        self, posterior: torch.Tensor, belief: torch.Tensor, observations: torch.Tensor
    ):
        """
        Assumes that the model has a defined decoder and reward model instantiated.

        Computes the reconstruction loss for the observation model (decoder)

        posterior and belief should be of shape:
            [horizon, batch_size, state_size]
            [horizon, batch_size, belief_size]

        observations should be of shape:
            [horizon, batch_size, *obs_size]
        """

        horizon, batch_size = observations.size(0), observations.size(1)

        # check if the posterior is the empty vector (RNN state space model)
        if posterior.nelement() != 0:
            posterior = posterior.view(horizon * batch_size, self._state_size)

        # check if the belief is the empty vector (SSM state space model)
        if belief.nelement() != 0:
            belief = belief.view(horizon * batch_size, self._belief_size)

        # mse will be of dimension: [horizon, batch_size, *self._obs_dim] 
        #(i.e. [horizon, batch_size, 3, 64, 64] in the case of pixel images.
        observation_pred = self.decode(posterior, belief).view(
            horizon, batch_size, *self._obs_dim
        )
        mse_obs = self.mse(observation_pred, observations)

        # sum over pixels/variables. The final n dimensions correspond to single obervation.
        # then mean over the horizon and batch dimension.
        mse_obs = mse_obs.sum(dim=[2 + i for i in range(len(self._obs_dim))])
        return mse_obs.mean(dim=(0, 1))

    def reward_loss(
        self, 
        reward_pred: torch.Tensor, 
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reward prediction loss."""
        reward_mse = self.mse(reward_pred, reward)
        return reward_mse.mean(dim=(0, 1))

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """
        pass observations through the image encoder.

        In the case of pixel images, the observations should be of shape
            [batch_size, color_channels, width, height]
        We enforce this in the first line by checking the dimension of the
        input data. We assume that is we have input of dimension 5, then it is of shape
            [horizon, batch_size, color_channels, width, height]
        and so we first flattent the input, and reshape it upon return.
        """

        # Check if the horizon dimension is present in the data. If so, flatten the observations.
        reshape = False
        size = observations.size()
        if len(self._obs_dim) + 1 < len(size):
            observations = observations.view(size[0] * size[1], *self._obs_dim)
            reshape = True

        embeddings = self._encoder(observations)

        # if we have flattened the input, we need to reshap the embeddings to be [horizon, batch_size, embed_size]
        if reshape:
            embeddings = embeddings.view(size[0], size[1], self._embed_size)

        return embeddings

    def observation_to_state_belief(self,
            prev_state,
            prev_action,
            prev_belief,
            observation: torch.Tensor
        ):
        """
        Helper function to encode first observation

        Args:
            obs_0 (torch.Tensor(1, *self._obs_size)): The first observation
        """

        observation = observation.to(self._device)
        belief, _, _, _, posterior_state, _, _, _ = self.forward(
            prev_state,
            prev_action,
            prev_belief,
            self.encode(observation).unsqueeze(dim=0)
        )
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(dim=0)
        return belief, posterior_state

    @abstractmethod
    def decode(self, posterior: torch.Tensor, belief: torch.Tensor):
        """pass the posterior state and belief state through the decoder."""
        raise NotImplementedError
