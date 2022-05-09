"""

Implimentation of the Deterministic State Space Model (RNN)

"""

from typing import Optional, List
from Models.base import TransitionModel
from torch import nn
import torch
import numpy as np

class RNN(TransitionModel, nn.Module):
    """
        Impliments the Deterministic State Space Model as Defined in the paper:
            Learning Latent Dynamics for Planning from Pixels, Hafner et. al. (2019)
            https://arxiv.org/pdf/1811.04551.pdf
         
        Model Outline:
            Transition Model: 
                Recurent Neural Network which takes in prior RNN state, h_{t}, and the last action, a_t, 
                and predicts the distribution, h_{t+1} ~ p(h_{s+t} | h_t, a_t). 
                This distribution is modeled as a Gaussian distribution with diagonal covariance.
            
            Posterior Model:
                Given the current observation (pixel level image, or the latent vector from an encoder network),
                the prior state of the RNN, h_{t-1}, from the RNN model, and the prior action, a_{t-1}, we can construct
                the posterior distribution for the current RNN hidden state, h_{t} ~ p(h_{t} | o_{t}, h_{t-1}, a_{t-1}). 
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
        super(RNN, self).__init__(  
            obs_dim, act_size, device, state_size, embed_size, belief_size, hidden_dim, activation, min_stddev
        )
        
        # Define the Transition model
        self._transition = nn.GRUCell(self._act_size, self._belief_size)

        # Define the Posterior model
        self._posterior = nn.Sequential(
            nn.Linear(self._belief_size + self._act_size + self._embed_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, self._belief_size)
        )

        # Define the Reward model
        self._reward = nn.Sequential(
            nn.Linear(self._belief_size, self._hidden_dim),
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
                We have access to the previos state (prev_state), and we want to generate a sequence of
                prior states; i.e. sample a sequence in the latent space.
                In this mode, non_terminals, and observations will be None, and so we do not generate 
                posterior values.

            Training:
                We DO have access to the ground truth observations as well as the non_terminal values.
                This is training mode, so we perform both generative modeling with the transition model,
                and inference using the posterior model.

            prev_state:     torch.empty()
            actions:        torch.Tensor[seq_length, batch_size, act_size] 
            prev_beliefs:   torch.Tensor[batch_size, belief_size] 
            observations:   torch.Tensor[seq_length, batch_size, embed_size] 
            non_terminals:  torch.Tensor[seq_length, batch_size, 1] 

            In generative mode, the batch dimension will be ommited.

            Returns:
                beliefs:            torch.Tensor[seq_length, batch_size, belief_size]
                prior_states:       torch.empty()       
                prior_means:        torch.Tensor[seq_length, batch_size, belief_size]
                prior_stds:         torch.empty()
                posterior_states:   torch.empty()
                posterior_means:    torch.Tensor[seq_length, batch_size, belief_size]
                posterior_stds:     torch.empty()
                rewards:            torch.Tensor[seq_length, batch_size]

        """

        horizon = actions.size(0) + 1 # plus 1 for the previous state

        # create empty lists to store the model predictions. Note that we cannot use
        # a single tensor of length horizon, as autograd does not back prop through inplace writes.
        prior_beliefs = [torch.empty(0)] * horizon
        posterior_beliefs = [torch.empty(0)] * horizon
        rewards = [torch.empty(0)] * horizon

        # the initial input to the model
        prior_beliefs[0] = prev_beliefs 
        posterior_beliefs[0] = prev_beliefs

        non_terminal = non_terminals if non_terminals is not None else torch.ones(horizon) 

        # predict a sequence off length action.size()
        for t in range(horizon-1):
            # If we have the observations, use the posterior belief as the model input . (the inferred RNN hidden state given the observation)
            # otherwise, use the belied state (this occurs when the model is performing generative modeling)
            current_belief = prior_beliefs[t] if observations is None else posterior_beliefs[t]
            current_belief = current_belief * non_terminal[t]

            # Forward pass through the RNN to get subsequent RNN state (belief state)
            prior_beliefs[t + 1] = self._transition(actions[t], current_belief)

            # if we have access to the observations, then we are in training mode, and we need to compute the posterior states.
            if observations is not None:
                # the posterior model takes the prior belief and action, as well as the current observation.
                posterior_input = torch.cat([observations[t], current_belief, actions[t]], dim=1)
                posterior_beliefs[t + 1] = self._posterior(posterior_input)

            rewards[t + 1] = self._reward(prior_beliefs[t+1] if observations is None else posterior_beliefs[t+1])
        
        # stack the list to convert to vextor, and remove redunct indices. Note thate the first element of the list is never update.
        prior_beliefs = torch.stack(prior_beliefs[1:], dim=0)
        posterior_beliefs = torch.stack(posterior_beliefs[1:],  dim=0)
        rewards = torch.stack(rewards[1:], dim=0).squeeze()

        # we return an empty tensor for compatiability with the recurrent states space models which require this value.
        return (
            prior_beliefs, 
            torch.empty(0).to(self._device), 
            prior_beliefs, 
            torch.empty(0).to(self._device), 
            torch.empty(0).to(self._device), 
            posterior_beliefs, 
            torch.empty(0).to(self._device), 
            rewards
        )


    def decode(self, latent: torch.empty, belief: torch.Tensor) -> torch.Tensor:
        """
            pass the latent state through the decoder. Ignore the latent vector, as this is only included for 
            cross compatability with the other Transition models.

            For RNN:
                latent: torch.empty (when using the RNN, there is no seperate latent state,)
                belief: torch.Tensor (Expected shape [horizon, batch_size, belief_size])
        """
        # check to see if we need to reshape the input.
        size = belief.size()
        reshape = False
        if 2 < len(size):
            belief = belief.view(size[0]*size[1], size[3])
            reshape = True

        observations = self._decoder(latent, belief)
        
        # if we flattened the input, we need to return to original dimenstions.
        if reshape:
            observations = observations.view(size[0], size[1], *self._obs_dim)
        return observations

    def kl_loss(
        self,
        prior_beliefs: torch.Tensor,
        place_holder1: torch.empty,
        posterior_beliefs: torch.Tensor,
        place_holder2: torch.empty,
        place_holder3: torch.Tensor,
    ) -> torch.Tensor:
        """
        For the RNN based model we do not have a stochastic component, and there is 
        therefore no distribution to learn. Instead, this is reduced to simply a MSE
        loss term between the prior_means and the posterior_means.

        prior_beliefs and posterior_beliefs should of the same dimension:
            [horizon, batch_size, belief_size]

        max_divergence is ignored in the computation of this loss term
        """
        
        # take the sum over the belief dimensiont
        mse_loss = self.mse(prior_beliefs, posterior_beliefs).sum(2)

        # return the mean loss accross the time and batch dimensions.
        return mse_loss.mean(dim=(0, 1))

