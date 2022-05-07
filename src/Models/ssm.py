"""

Implimentation of the Gaussian Stochastic State Space Model (SSM)

"""

from typing import Optional, List
from Models.base import TransitionModel
from torch import nn
import torch
import numpy as np

class SSM(TransitionModel, nn.Module):
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
        super(SSM, self).__init__(  
            obs_dim, act_size, device, state_size, embed_size, belief_size, hidden_dim, activation, min_stddev
        )
        
        # Define the Transition model
        self._transition = nn.Sequential(
            nn.Linear(self._state_size + self._act_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Posterior model
        self._posterior = nn.Sequential(
            nn.Linear(self._state_size + self._act_size + self._embed_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 2 * self._state_size)
        )

        # Define the Reward model
        self._reward = nn.Sequential(
            nn.Linear(self._state_size, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            self._activation(),
            nn.Linear(self._hidden_dim, 1)
        )

    def forward(
        self, 
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_beliefs: Optional[torch.Tensor] = None, 
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

            prev_state:     torch.Tensor[seq_length, batch_size, state_size] 
            actions:        torch.Tensor[seq_length, batch_size, act_size] 
            prev_beliefs:   torch.Tensor[seq_length, batch_size, belief_size] 
            observations:   torch.Tensor[seq_length, batch_size, embed_size] 
            non_terminals:  torch.Tensor[seq_length, batch_size, 1] 

            In generative mode, the batch dimension will be ommited.

            Returns:
                beliefs:            For the SSM this is simply the empty tensor. 
                prior_states:       torch.Tensor[seq_length, batch_size, state_size] 
                prior_means:        torch.Tensor[seq_length, batch_size, state_size] 
                prior_stds:         torch.Tensor[seq_length, batch_size, state_size] 
                posterior_states:   torch.Tensor[seq_length, batch_size, state_size] 
                posterior_means:    torch.Tensor[seq_length, batch_size, state_size] 
                posterior_stds:     torch.Tensor[seq_length, batch_size, state_size] 
                rewards:            torch.Tensor[seq_length, batch_size]

        """

        horizon = actions.size(0) + 1 # plus 1 for the previous state

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
            prior_means[t + 1], prior_log_stddvs = torch.chunk(self._transition(prior_input), 2, dim = 1)
            prior_stddvs[t + 1] = self.softplus(prior_log_stddvs) + self._min_stddev
            prior_states[t + 1] = prior_means[t+1] + torch.randn_like(prior_means[t+1]) * prior_stddvs[t + 1] 

            # if we have access to the observations, then we are in training mode, and we need to compute the posterior states.
            if observations is not None:
                # the posterior model takes the prior state and action, as well as the current observation.
                posterior_input = torch.cat([observations[t], current_state, actions[t]], dim=1)
                posterior_means[t + 1], posterior_log_stddvs = torch.chunk(self._posterior(posterior_input), 2, dim = 1)
                posterior_stddvs[t + 1] = self.softplus(posterior_log_stddvs) + self._min_stddev
                posterior_states[t + 1] = posterior_means[t+1] + torch.randn_like(posterior_means[t+1]) * posterior_stddvs[t + 1] 

            rewards[t + 1] = self._reward(prior_states[t+1] if observations is None else posterior_states[t+1])
        
        # stack the list to convert to vextor, and remove redunct indices. Note thate the first element of the list is never update.
        prior_states = torch.stack(prior_states[1:], dim=0)
        prior_means = torch.stack(prior_means[1:], dim=0)
        prior_stddvs = torch.stack(prior_stddvs[1:],  dim=0)
        posterior_states = torch.stack(posterior_states[1:],  dim=0)
        posterior_means = torch.stack(posterior_means[1:],  dim=0)
        posterior_stddvs = torch.stack(posterior_stddvs[1:],  dim=0)
        rewards = torch.stack(rewards[1:], dim=0).squeeze()

        # we return an empty tensor for compatiability with the recurrent states space models which require this value.
        return torch.empty(0).to(self._device), prior_states, prior_means, prior_stddvs, posterior_states, posterior_means, posterior_stddvs, rewards


    def decode(self, latent: torch.Tensor, belief: torch.empty) -> torch.Tensor:
        """
            pass the latent state through the decoder. Ignore the belief vector, as this is only included for 
            cross compatability with the other Transition models.

            For SSM:
                latent: torch.Tensor with dimensions [horizon, batch_size, state_size]
                belief: when using the SSM, there is no belief state, so this should 
                        always be torch.empty(0). This is because
                            torch.cat([latent, belief], dim=1) 
                        simply returns the latent vector, since concatentating the 
                        empty tensor leaves the other tensor unchanged.
        """
        # check to see if we need to reshape the input.
        size = latent.size()
        reshape = False
        if 2 < len(size):
            latent = latent.view(size[0]*size[1], size[3])
            reshape = True

        observations = self._decoder(latent, belief)
        
        # if we flattened the input, we need to return to original dimenstions.
        if reshape:
            observations = observations.view(size[0], size[1], *self._obs_dim)
        return observations

    def forward_generate(self, batched_actions, obs_0):
        """Helper function to generate rewards / states

        TODO: Currently this function only accepts one observation for t0
        - It should be able to support different t0 states for the different actions
        - It should be able to support variable length t0_actions, t0_observations

        Args:
            batched_actions (torch.Tensor[seq_length, batch_size, action_size]): Actions to pass forward through the model
            obs_0 (torch.Tensor[1, *self._state_size]): Observation for the t0 state
        """

        if not len(obs_0.shape) > 3:
            obs_0 = obs_0.unsqueeze(0)
        assert obs_0.shape[1:] == self._obs_dim

        # Call forward once with observations to get posterior_state
        # Starting with actions, init_belief, init_state as zero
        # Actions will be used in the next call
        seq_length, batch_size, action_size = batched_actions.shape
        init_belief = torch.zeros(1, 0)
        init_state = torch.zeros(1, self._state_size)
        init_action = torch.zeros(1, 1, self._act_size)
        encoded_observation_0 = self.encode(obs_0)
        if not len(encoded_observation_0.shape) >= 3:
            encoded_observation_0 = encoded_observation_0.unsqueeze(0)
        (
            t0_beliefs,
            t0_prior_states,
            t0_prior_means,
            t0_prior_stddvs,
            t0_posterior_states,
            t0_posterior_means,
            t0_posterior_stddvs,
            t0_reward_preds,
        ) = self.forward(
            init_state,
            init_action,
            init_belief,
            encoded_observation_0
        )

        # t0 state is repeated along batch dimension so all actions have the same starting t0
        t0_posterior_states = torch.concat([t0_posterior_states[0]]*batch_size, dim=0)

        # Call forward again with no observations but prev_state
        (
            generated_beliefs,
            generated_prior_states,
            generated_prior_means,
            generated_prior_stddvs,
            generated_posterior_states,
            generated_posterior_means,
            generated_posterior_stddvs,
            generated_reward_preds,
        ) = self.forward(
            t0_posterior_states,
            batched_actions
        )

        reconstructed_obs = self._decoder(generated_prior_states, generated_beliefs)
        reconstructed_images = reconstructed_obs.permute(0, 2, 3, 1).detach().numpy()
        generated_rewards = generated_reward_preds.detach().numpy()
        return generated_rewards, reconstructed_images

        
