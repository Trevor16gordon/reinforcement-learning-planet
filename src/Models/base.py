"""

Implimentation of the Gaussian Stochastic State Space Model (SSM)

"""

from typing import Optional, List
from abc import ABC, abstractmethod

from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


import numpy as np

class TransitionModel(ABC):
    """
        Base Class for all Transition Models. 

        All inheriting classes should support the following methods:

            encode: pass pixel/symbolic level observations through the encoder.
            decode: pass the latent state and belief state through the decoder.
            forward: standard foward pass through the transition model.
            loss: Compute the model loss using the KL Divergence loss, the reconstruction loss, and the reward loss.
    """

    def __init__(self, 
        encoder,
        decoder,
        obs_dim,
        state_size,
        act_size,
        embed_size,
        belief_size,
        hidden_dim,
        activation=nn.elu, 
        min_stddev=1e-5
    ):
        self._encoder = encoder
        self.decoder = decoder
        self._obs_dim = obs_dim

        self._state_size = state_size 
        self._act_size = act_size
        self._embed_size = embed_size
        self._hidden_dim = hidden_dim
        self._activation = activation
        self._min_stddev = min_stddev

        self.mse = nn.MSELoss(reduction="none")

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

        reshape = None

        # Check if the horizon dimension is present in the data. If so, flatten the observations.
        if len(self._obs_dim) + 1 < len(observations.size()):
            reshape = [observations.size()[0], observations.size()[1]]
            observations = observations.view(reshape[0]*reshape[1], *self._obs_dim) 

        embeddings = self._encoder(observations)
        
        # if we have flattened the input, we need to reshap the embeddings to be [horizon, batch_size, embed_size]
        if reshape is not None:
            embeddings = embeddings.view(reshape[0], reshape[1], self._embed_size) 

        return embeddings 

    def kl_loss(self, 
            prior_means: torch.Tensor, 
            prior_stddev: torch.Tensor, 
            posterior_means: torch.Tensor, 
            posterior_stddev: torch.Tensor,
            max_divergence: torch.Tensor,
    ) -> torch.Tensor:
        """
            Compute the full model Loss function.

            prior_means, prior_stddev, posterior_means, posterior_stddev should
            all be of the same dimension:
                [horizon, batch_size, state_dim]
            
            max_divergence is an upper bound on the KL_divergence loss, and is used to 
            help stabalize training.
        """

        prior_dist = Normal(prior_means, prior_stddev)
        posterior_dist = Normal(posterior_means, posterior_stddev)
        kl_loss = kl_divergence(prior_dist, posterior_dist)

        # sum accross the state dimension, and clip the loss. 
        kl_loss = torch.max(kl_loss.sum(dim=2), max_divergence)

        # return the mean loss accross the time and batch dimensions.
        return kl_loss.mean(dim=(0,1))

    def observation_loss(
        posterior: torch.Tensor,
        belief: torch.Tensor,
        observations: torch.Tensor
        rewards: torch.Tensor
    ):
    """
        Assumes that the model has a defined decoder and reward model instantiated.
    
        Computes the reconstruction loss for the observation model (decoder) 

        posterior and belief should be of shape:
    """

        observation_pred = self.



    @abstractmethod
    def decode(self, posterior: torch.Tensor, belief: torch.Tensor):
        """
            pass the posterior state and belief state through the decoder.
        """
        raise NotImplementedError
