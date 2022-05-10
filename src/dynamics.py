""" 

Classes relating to the dynamics 
"""
import gym
import numpy as np
import pdb
import torch
from env import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class DynamicsBase():
    """Base class for environment dynamics stepping

    - Intended to be subclassed so that Dynamics with true and learned simulators can be used
    - All operations are intended to be vectorized
    """

    def __init__(self) -> None:
        return

    def advance(self, state, action):
        raise NotImplementedError()

    def advance_multiple_timesteps(self, state_0, action):
        raise NotImplementedError()


class TrueDynamics():

    def __init__(self, env_name):
        super(TrueDynamics, self).__init__()
        self.env_name = env_name
        dummy_env = GymEnv(env_name,  0, False, 30, 1, 8)
        self.env_action_dim = dummy_env.action_size
        self.env_state_dim = dummy_env.observation_size

    def advance(self, state, actions):
        """Advance a step in the dynamics

        Args:
            state (np.array): Shape should be (self.num_env, self.env_state_dim)
            action (np.array): Shape should be (self.num_env, self.env_action_dim)

        Returns:
            _type_: _description_
        """
        pass

    def advance_multiple_timesteps(self, state_0, actions_multiple_timesteps):
        """Advance multiple steps in the dynamics

        Args:
            actions_multiple_timesteps (np.array): Shape should be (num_timesteps, self.num_env,  self.env_action_dim)
            action (np.array): Shape should be (self.num_env, self.env_action_dim)
            state_0 (np.array): Shape should be (self.num_env, self.env_state_dim)

        Returns:
            next_states (np.array): Shape is (num_timesteps, self.num_env, self.env_state_dim)
            rewards (np.array): Shape is (num_timesteps, self.num_env, 1)
            dones (np.array): Shape is (num_timesteps, self.num_env, 1)
        """

        # TODO: Need to make this work for variable unflattened observation dimension
        num_timesteps, num_envs, env_action_dim_in = actions_multiple_timesteps.shape
        num_env_state0, env_state_dim = state_0.shape
        assert num_envs == num_env_state0
        h = env_state_dim // 2
        #envs = SubprocVecEnv([lambda : gym.make(self.env_name) for i in range(num_envs)], start_method="fork")
        envs = DummyVecEnv([lambda : gym.make(self.env_name) for i in range(num_envs)])
        envs.reset()

        # Setting state works if it's all the same arguments
        envs.env_method("set_state", state_0[0, :h], state_0[0, h:])

        resulting_next_states = np.zeros((num_timesteps, num_envs, env_state_dim))
        resulting_rewards = np.zeros((num_timesteps, num_envs))
        resulting_dones = np.zeros((num_timesteps, num_envs))

        for j in range(num_timesteps):
            actions = actions_multiple_timesteps[j, :, :]
            envs.step_async(actions)
            next_states, rewards, dones, _ = envs.step_wait()
            resulting_next_states[j, :, :] = next_states
            resulting_rewards[j, :] = rewards
            resulting_dones[j, :] = dones
        envs.close()
        return resulting_next_states, resulting_rewards, resulting_dones


class LearnedDynamics():

    def __init__(self, env_name, transition_model, action_size, observation_size):
        super(LearnedDynamics, self).__init__()
        self.env_name = env_name
        self.transition_model = transition_model
        self.model_belief = None
        self.model_state = None
        self.env_action_dim = action_size
        self.env_state_dim = observation_size

    def advance_multiple_timesteps(self, state_0, actions_multiple_timesteps):
        """Advance multiple steps in the dynamics

        Args:
            actions_multiple_timesteps (np.array): Shape should be (num_timesteps, self.num_env,  self.env_action_dim)
            action (np.array): Shape should be (self.num_env, self.env_action_dim)
            state_0 (np.array): Shape should be (self.num_env, self.env_state_dim)

        Returns:
            next_states (np.array): Shape is (num_timesteps, self.num_env, self.env_state_dim)
            rewards (np.array): Shape is (num_timesteps, self.num_env, 1)
            dones (np.array): Shape is (num_timesteps, self.num_env, 1)
        """
        state_0 = torch.from_numpy(state_0).float()
        #actions_multiple_timesteps = torch.from_numpy(actions_multiple_timesteps).float()
        _, _, _, _, _, _, _, generated_rewards = self.transition_model(self.model_state, actions_multiple_timesteps, self.model_belief)
        #generated_rewards = generated_rewards.cpu().detach().numpy()
        resulting_dones = None
        return None, generated_rewards, resulting_dones

class ModelPredictiveControl():

    def __init__(self, dynamics, min_action_clip=None, max_action_clip=None, cost_func=None):
        """

        Args:
            cost_func (func): Should take in states rewards, dones and determine a scoring for the resulting action
            dynamics (DynamicsBase): Used for rolling out action sequences. Dynamics should return flattened states
        """
        if cost_func == None:
            self.cost_func = cost_func if cost_func is not None else self.cost_func_default
        elif cost_func == "cost_func_closest_goal":
            self.cost_func = self.cost_func_closest_goal
        elif cost_func == "cost_func_avg_closest_goal":
            self.cost_func = self.cost_func_avg_closest_goal
        self.dynamics = dynamics
        self.control_horizon_simulate = 14
        self.min_action_clip = min_action_clip
        self.max_action_clip = max_action_clip

    def compute_action(self, state, goal):
        return self.compute_action_cross_entropy_method(state, goal)

    def compute_action_cross_entropy_method(self, state, goal, num_iterations=3, j=50, k=10):

        env_action_dim = self.dynamics.env_action_dim
        num_timesteps = self.control_horizon_simulate

        # Create J candidate sequences
        device = self.dynamics.transition_model._device

        # Ok I shouldn't do this reshaping here
        # Need to deal with flatenned state dimension as well as unflattened like images (3, 64, 64)
        state0_duplicates = np.repeat(np.expand_dims(state, 0), j, axis=0)

        #means = np.zeros((1, self.control_horizon_simulate))
        #stds = 1*np.ones((1, self.control_horizon_simulate))
        means = torch.zeros(self.control_horizon_simulate, 1, device=device)
        stds = torch.ones(self.control_horizon_simulate, 1, device=device)
        means = means.unsqueeze(1)
        stds = stds.unsqueeze(1)
        for iter in range(num_iterations):
            

            means = means.repeat(1, j, 1)
            stds = stds.repeat(1, j, 1)
            candidate_actions = means + stds * torch.randn(num_timesteps, j, env_action_dim, device=device)
            #candidate_actions = np.random.normal(loc=means, scale=stds, size=(j,  env_action_dim, num_timesteps))
            #candidate_actions = np.swapaxes(candidate_actions, 0, 2)
            #candidate_actions = np.swapaxes(candidate_actions, 1, 2)
            if (self.min_action_clip is not None) and (self.max_action_clip is not None):
                candidate_actions.clamp(self.min_action_clip, self.max_action_clip)
            resulting_next_states, resulting_rewards, resulting_dones = self.dynamics.advance_multiple_timesteps(state0_duplicates, candidate_actions)
            costs = self.cost_func(resulting_next_states, resulting_rewards, resulting_dones, goal)
            costs_k, ids = torch.topk(costs, k, dim=1, largest=False)
            ids = ids.squeeze()
            top_action_seqs = torch.index_select(candidate_actions, 1, ids)
            #top_action_seqs = candidate_actions[:, ids, :]
            # Calculate the mean and variance of the resulting.
            #means = np.mean(top_action_seqs, axis=1).reshape((1, -1))
            #stds = np.std(top_action_seqs, axis=1).reshape((1, -1))
            means = torch.mean(top_action_seqs, dim=1, keepdim=True)
            stds = torch.std(top_action_seqs, dim=1, keepdim=True)
        #best_action_seq = candidate_actions[:, ids[0], :]
        means = means.squeeze(1)
        ret = means.cpu().detach().numpy()
        return ret

    def cost_func_default(self, next_states, rewards, dones, goal):
        """Default reward function that simply aggregates the total rewards

        This module is setupt to minimize the cost. So we multiply the rewards as negative
        Args:
            next_states (np.array): Shape is (num_timesteps, self.num_env, self.env_state_dim)
            rewards (np.array): Shape is (num_timesteps, self.num_env, 1)
            dones (np.array): Shape is (num_timesteps, self.num_env, 1)
            goal (_type_): _description_
        
        Returns
            rewards (np.array) Shape is (num_envs), 1
        """
        #return -1*np.sum(rewards, axis=0)
        return -1 * torch.sum(rewards, dim=0, keepdim=True)

    def cost_func_closest_goal(self, next_states, rewards, dones, goal):
        """Returns the euclidian distance to the goal from the final state

        Args:
            next_states (np.array): Shape is (num_timesteps, self.num_env, self.env_state_dim)
            rewards (np.array): Shape is (num_timesteps, self.num_env, 1)
            dones (np.array): Shape is (num_timesteps, self.num_env, 1)
            goal (_type_): _description_
        
        Returns
            rewards (np.array) Shape is (num_envs), 1
        """
        last_state = next_states[-1, :, :]
        return np.linalg.norm(last_state - goal, axis=1)

    def cost_func_avg_closest_goal(self, next_states, rewards, dones, goal):
        """Returns the euclidian distance to the goal averaged over all states

        Args:
            next_states (np.array): Shape is (num_timesteps, self.num_env, self.env_state_dim)
            rewards (np.array): Shape is (num_timesteps, self.num_env, 1)
            dones (np.array): Shape is (num_timesteps, self.num_env, 1)
            goal (_type_): _description_
        
        Returns
            rewards (np.array) Shape is (num_envs), 1
        """
        ret = np.linalg.norm(np.mean(next_states[:, :, :2]- goal[:2], axis=0), axis=1)
        return ret



if __name__ == "__main__":

    # dyn = TrueDynamics("InvertedPendulum-v2")
    
    

    mpc = ModelPredictiveControl(dyn)
    mpc.control_horizon_simulate = 3

    state0 = np.zeros((4,1))
    goal = np.array([0, 0, 0, 0])

    # best_action = mpc.compute_action_cross_entropy_method(state0, goal, num_iterations=10, j=30, k=10)


    env = gym.make("InvertedPendulum-v2")
    env.reset()

    num_to_rollout_at_once = 10

    nframes = 30
    state = env.reset()
    for i in range(nframes):
        #renders the environment
        env.render()
        best_actions = mpc.compute_action_cross_entropy_method(state, goal, num_iterations=5, j=100, k=10)

        for j in range(num_to_rollout_at_once):
            action = best_actions[j, :]
            next_state, rewards, done, info  = env.step(action)
        
        state = next_state
