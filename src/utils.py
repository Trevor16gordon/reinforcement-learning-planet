""" 

This file is intended to be a catch all for misc functions

"""

import numpy
from data import ExperienceReplay

def gather_data(env, memory: ExperienceReplay, n_trajectories: int = 5) -> None:
    """
        Gather N trajectories using random actions and add the transitions to the 
        experience replay memory.
    """
    
    for _ in range(n_trajectories):
        state = env.reset()
        done = False
        while not done: 
            action = env.sample_random_action()
            print(action, state)
            next_state, reward, done, info  = env.step(action)
            memory.append(next_state, action, reward, done)
 
