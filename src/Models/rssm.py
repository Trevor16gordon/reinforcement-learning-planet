"""

Implimentation of the Full Recurrent State Space Model

"""

from typing import Optional, List
from Models.base import TransitionModel
from torch import nn
import torch
import numpy as np

class RSSM(TransitionModel, nn.Module):
    """
        Impliments the Recurrent State Space Model from the paper:
            Learning Latent Dynamics for Planning from Pixels, Hafner et. al. (2019)
            https://arxiv.org/pdf/1811.04551.pdf
         
        Model Outline:
            Recurrent Model:
                This is a GRU cell which is used to retainn historical information regarding the state 
                space dynamics. This violates the Markovian modelling assumption of the SSM model, and
                allows it to apply to more complex domains. The model takes in the prior hidden state and 
                action, h_{t} & a_{t}, and predictis the new hidden state, h_{t+1}

            Transition Model: 
                Feed Forward Network which takes in the prior belief state, h_{t}, and predicts 
                the distribution, s_{t+1} ~ p(s_{s+t} | h_{t}).This distribution is 
                modeled as a Gaussian distribution with diagonal covariance.
            
            Posterior Model:
                Given the current observation (pixel level image, or the latent vector from an encoder network),
                and the belief state, h_{t-1}, from the RNN model, we can construct the posterior distribution 
                for the current latent state, s_{t} ~ p(s_{t} | o_{t}, h_{t}). This is used as the learning target
                for the Transition Model output. 

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
        obs_dim,
        act_size,
        device,
        state_size,
        embed_size,
        belief_size,
        hidden_dim,
        activation="ELU", 
        min_stddev=1e-1
    ):
        """
            obs_dim:     The dimenstion of the observation space. Used to validate encoder input.
            act_size:    The dimension of the action space. (Assumed to be continuous, but can also use one-hot vectors for discrete action spaces.)
            device:      The device that the model is running on (cpu or gpu)
            state_size:  The size of the latent state vector.
            embed_size:  The size of the embedding layer from the encoder and decoder.  
            hidden_size: The width of the hidden layers in the networks.
            belief_size: Dimension of hidden state. NOT USED FOR SSM.
            activation:  The activation function used in the networks.
            min_stddev:  lower bound on the predicted latent state variance.

            we include the belief size for compatability with the Recurrent state space models, however it is unused.
        """

        # Call super from the base class to instantiate all model hyper parameters. Then call nn.Module to set up internal models
        nn.Module.__init__(self)
        super(RSSM, self).__init__(  
            obs_dim, act_size, device, state_size, embed_size, belief_size, hidden_dim, activation, min_stddev
        )
         
        # Define the RNN model, we include a sindle embedding layer to go from 
        # self._state_size + self._action_size -> self.belief_size
        self._embed_state_act = nn.Sequential(
            nn.Linear(self._state_size + self._act_size, self._belief_size),
            self._activation()
        )
        self._rnn = nn.GRUCell(self._belief_size, self._belief_size)

        # Define the Transition model
        self._transition = nn.Sequential(
            nn.Linear(self._belief_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Posterior model
        self._posterior = nn.Sequential(
            nn.Linear(self._belief_size +  self._embed_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Reward model
        self._reward = nn.Sequential(
            nn.Linear(self._state_size + self._belief_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 1)
        )

    def forward(
        self, 
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_beliefs: torch.Tensor, 
        observations: Optional[torch.Tensor] = None,
        non_terminals: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
            Forward operates in two modes:
            Generative:
                We have access to the previos state (prev_state) and belief (RNN hidden state), and
                we want to generate a sequence of prior states; i.e. sample a sequence in the latent space.
                In this mode, non_terminals, and observations will be None, and so we do not generate 
                posterior values.

            Training:
                We DO have access to the ground truth observations as well as the non_terminal values.
                This is training mode, so we perform both generative modeling with the transition model,
                and inference using the posterior model.

            prev_state:     torch.Tensor[batch_size, state_size] 
            actions:        torch.Tensor[seq_length, batch_size, act_size] 
            prev_beliefs:   torch.Tensor[batch_size, belief_size] 
            observations:   torch.Tensor[seq_length, batch_size, embed_size] 
            non_terminals:  torch.Tensor[seq_length, batch_size, 1] 

            In generative mode, the batch dimension will be ommited.

            Returns:
                beliefs:            torch.Tensor[seq_length, batch_size, belief_size]
                prior_states:       torch.Tensor[seq_length, batch_size, state_size] 
                prior_means:        torch.Tensor[seq_length, batch_size, state_size] 
                prior_stds:         torch.Tensor[seq_length, batch_size, state_size] 
                posterior_states:   torch.Tensor[seq_length, batch_size, state_size] 
                posterior_means:    torch.Tensor[seq_length, batch_size, state_size] 
                posterior_stds:     torch.Tensor[seq_length, batch_size, state_size] 
                rewards:            torch.Tensor[seq_length, batch_size]

        """

        horizon = actions.size(0) + 1 # plus 1 for the previous state
        
        # empty list to store RNN hidden states.
        beliefs = [torch.empty(0)] * horizon

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
        
        # initial belief will be the given prev_belief value
        beliefs[0] = prev_beliefs

        non_terminal = non_terminals if non_terminals is not None else torch.ones(horizon) 

        # predict a sequence off length action.size()
        for t in range(horizon-1):
            # If we have the observations, use the posterior state as the input to the transition model. (the inferred state from prior observation)
            # otherwise, use the prior state (this occurs when the model is performing generative modeling)
            current_state = prior_states[t] if observations is None else posterior_states[t]
            current_state = current_state * non_terminal[t]

            # We need to also zero out the RNN state if we are starting a new trajectory.
            prior_belief = beliefs[t] * non_terminal[t]
            
            # compute the next belief state of the RNN. 
            state_actions = torch.cat([current_state, actions[t]], dim=1)
            state_act_embedding = self._embed_state_act(state_actions)
            beliefs[t+1] = self._rnn(state_act_embedding, prior_belief) 

            # compute prior distribution of next state
            prior_means[t + 1], prior_log_stddvs = torch.chunk(self._transition(beliefs[t + 1]), 2, dim = 1)
            prior_stddvs[t + 1] = self.softplus(prior_log_stddvs) + self._min_stddev
            prior_states[t + 1] = prior_means[t+1] + torch.randn_like(prior_means[t+1]) * prior_stddvs[t + 1] 

            # if we have access to the observations, then we are in training mode, and we need to compute the posterior states.
            if observations is not None:
                # the posterior model takes the prior state and action, as well as the current observation.
                posterior_input = torch.cat([observations[t], beliefs[t+1]], dim=1)
                posterior_means[t + 1], posterior_log_stddvs = torch.chunk(self._posterior(posterior_input), 2, dim = 1)
                posterior_stddvs[t + 1] = self.softplus(posterior_log_stddvs) + self._min_stddev
                posterior_states[t + 1] = posterior_means[t+1] + torch.randn_like(posterior_means[t+1]) * posterior_stddvs[t + 1] 
            
            # Use posterior states to predict rewards if the are available. Otherwise use the prior states.
            rew_state = prior_states[t+1] if observations is None else posterior_states[t+1]
            reward_input = torch.cat([rew_state, beliefs[t+1]], dim=1)
            rewards[t + 1] = self._reward(reward_input)
        
        # stack the list to convert to vextor, and remove redunct indices. Note thate the first element of the list is never update.
        beliefs = torch.stack(beliefs[1:], dim=0)
        prior_states = torch.stack(prior_states[1:], dim=0)
        prior_means = torch.stack(prior_means[1:], dim=0)
        prior_stddvs = torch.stack(prior_stddvs[1:],  dim=0)
        posterior_states = torch.stack(posterior_states[1:],  dim=0)
        posterior_means = torch.stack(posterior_means[1:],  dim=0)
        posterior_stddvs = torch.stack(posterior_stddvs[1:],  dim=0)
        rewards = torch.stack(rewards[1:], dim=0).squeeze()

        # we return an empty tensor for compatiability with the recurrent states space models which require this value.
        return beliefs, prior_states, prior_means, prior_stddvs, posterior_states, posterior_means, posterior_stddvs, rewards


    def decode(self, latent: torch.Tensor, belief: torch.empty) -> torch.Tensor:
        """
            pass the latent state through the decoder. Ignore the belief vector, as this is only included for 
            cross compatability with the other Transition models.

            For RSSM:
                latent: torch.Tensor with dimensions [horizon, batch_size, state_size]
                belief: torch.Tensor with dimensions [horizon, batch_size, belief_size]
        """
        # check to see if we need to reshape the input.
        size = latent.size()
        reshape = False
        if 2 < len(size):
            latent = latent.view(size[0]*size[1], -1)
            belief = belief.view(size[0]*size[1], -1)
            reshape = True

        observations = self._decoder(latent, belief)
        
        # if we flattened the input, we need to return to original dimenstions.
        if reshape:
            observations = observations.view(size[0], size[1], *self._obs_dim)
        return observations
