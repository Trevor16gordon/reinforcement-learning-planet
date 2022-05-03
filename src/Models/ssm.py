"""

Implimentation of the Gaussian Stochastic State Space Model (SSM)

"""

from typing import Optional, List
from torch import nn
import torch
import numpy as np


class SSM(nn.Module)
    """
        Impliments the Gaussian Stochastic State Space Model as Defined in the paper:
            Learning Latent Dynamics for Planning from Pixels, Hafner et. al. (2019)
            https://arxiv.org/pdf/1811.04551.pdf
         
        Model Outline:
            Transition Model: 
                Feed Forward Network which takes in prior latent state, s_t, and the last action, a_t, 
                and predicts the distribution, s_{t+1} ~ p(s_{s+t} | s_t, a_t). 
                This distribution is modeled as a Gaussian distribution with diagonal covariance.
            
            Posterior Model:
                Given the current observation (pixel level image, or the latent vector from an encoder network),
                the prior latent state, s_{t-1}, from the SSM model, and the prior action, a_{t-1}, we can construct
                the posterior distribution for the current latent state, s_{t} ~ p(s_{t} | o_{t}, s_{t-1}, a_{t-1}). 
                This is used as the learning target for the Transition Model output. 

        Loss:
            Similar to the other transition model, we have two loss terms,
            KL-Divergence between the posterior distribution, and the distribution predicted by the transition model.

            Reconstruction Error: This is the loss term coming from the VAE component of the latent space model. Sample 
            from the posterior distribution and pass this through a decoder network, which provides a reconstruction
            error from the base image.

            Reward Error: An \ell_{2} loss on predicting the reward from the latent state observations.

    """

    def __init__(
        self,
        state_size,
        act_size,
        embed_size,
        belief_size,
        hidden_dim,
        activation=nn.elu, 
        min_stddev=1e-5
    ):
        """
            state_size: The size of the latent state vector.
            actOsize: The dimension of the action space. (Assumed to be continuous, but can also use one-hot vectors for discrete action spaces.)
            embed_size: The size of the embedding layer from the encoder and decoder.  
            hidden_size: The width of the hidden layers in the network.

            we include the belief size for compatability with the Recurrent state space models, however it is unused.
        """

        self._state_size = state_size 
        self._act_size = act_size
        self._embed_size = embed_size
        self._hidden_dim = hidden_dim
        self._activation = activation
        self._min_stddev = min_stddev

        # Define the Transition model
        self._transition = nn.Sequential(
            nn.Linear(self._state_size + self_act_size, self._hidden_dim),
            self._activation()
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Posterior model
        self._posterior = nn.Sequential(
            nn.Linear(self._state_size + self_act_size + self._embed_size, self._hidden_dim),
            self._activation()
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Reward model
        self._reward = nn.Sequential(
            nn.Linear(self._state_size, self._hidden_dim),
            self._activation()
            nn.Linear(self._hidden_dim, self._hidden_dim)
            self._activation()
            nn.Linear(self._hidden_dim, 1)
        )

        self.softplus = nn.Softplus()


    def forward(self, 
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_beliefs: Optional[torch.Tensor] = None, 
        observations: Optional[torch.Tensor]=None,
        non_terminals: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
            Forward operates in two modes:
                Generative:
                    We have access to the previos state (prev_state), and we want to generate a sequence of
                    prior states; i.e. sample a sequence in the latent space.
                    In this mode, non_terminals, and observations will be None, and so we do not generate 
                    posterior values.

                Training:
                    We DO have access to the ground truth observations as well as the non_terminal values.
                    This is training mode, so we perform both generative modeling with the transition model,
                    and inference using the posterior model.

            pre_state:      torch.Tensor[seq_length, batch_size, state_size] 
            actions:        torch.Tensor[seq_length, batch_size, act_size] 
            observations:   torch.Tensor[seq_length, batch_size, embed_size] 
            non_terminals:  torch.Tensor[seq_length, batch_size, 1] 

            In generative mode, the batch dimension will be ommited.
        """
    
        horizon = actions.size[0] + 1 # plus 1 for the previous state
        
        # create empty lists to store the model predictions. Note that we cannot use
        # a single tensor of length horizon, as autograd does not back prop through inplace writes.
        prior_states = [torch.empty(0)] * horizon
        prior_means = [torch.empty(0)] * horizon
        prior_stddvs = [torch.empty(0)] * horizon

        posterior_states = [torch.empty(0)] * horizon
        posterior_means = [torch.empty(0)] * horizon
        posterior_stddvs = [torch.empty(0)] * horizon

        rewards = [torch.empty(0)] * horizon

        # the first prior and posterior state will be the previous observations
        prior_states[0] = prev_state
        posterior_states[0] = prev_state
        
        non_terminal = non_terminals if non_terminals is not None else torch.ones(horizon) 

        # predict a sequence off length action.size()
        for t in range(horizon-1):

            # If we have the observations, use the posterior state as the input to the transition model. (the inferred state from prior observation)
            # otherwise, use the prior state (this occurs when the model is performing generative modeling)
            current_state = prior_states[t] if observations is None else posterior_states[t]
            current_state = current_state * non_terminal[t]

            # compute prior distribution of next state
            prior_input = torch.cat([current_state, actions[t]], dim=1)
            prior_means[t + 1], prior_stddvs = torch.chunk(self._transition(prior_input), 2, dim = 1)
            prior_stddvs[t + 1] = self.softplus(prior_stddvs) + self._min_stddev
            prior_states[t + 1] = prior_means[t+1] + torch.randn_like(prior_means[t+1]) * prior_stddvs[t + 1] 

            # if we have access to the observations, then we are in training mode, and we need to compute the posterior states.
            if observations is not None:
                # the posterior model takes the prior state and action, as well as the current observation.
                posterior_input = torch.cat([observations[t], current_state, actions[t]], dim=1)
                posterior_means[t + 1], posterior_stddvs = torch.chunk(self._transition(posterior_input), 2, dim = 1)
                posterior_stddvs[t + 1] = self.softplus(posterior_stddvs) + self._min_stddev
                posterior_states[t + 1] = posterior_means[t+1] + torch.randn_like(posterior_means[t+1]) * posterior_stddvs[t + 1] 

            rewards[t + 1] = self._reward(prior_states[t+1] if observations is None else posterior_states[t+1])
        
        # we return an empty tensor for compatiability with the recurrent states space models which require this value.
        return torch.empty(0), prior_states, prior_means, prior_stddvs, posterio_states, posterion_means, posterior_stddvs, rewards


