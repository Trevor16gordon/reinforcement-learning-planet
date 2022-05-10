# PlaNet: Learning Latent Dynamics for Planning from Pixels

This repo contains a pytorch implementation and study of the origiinal Google paper


Planing with known environment dynamics is a highly effective way to solve complex control problems. However, for unobserved environments, we must utilize observations of agent interaction to learn models of the world. 
Deep Planning Network (PlaNet) is an approach that learns the approximate environment dynamics from images, and chooses actions through fast online planning in latent space.
PlaNet uses a latent dynamics model, which contains both deterministic and stochastic transition components, to solve continuous control tasks which exceed the difficulty of tasks previously solved similar methods.
PlaNet is also incredibly data-efficient, and outperforms model-free methods final performance, with on average 200×  fewer environment interactions and similar computation time.


# Key Results

Below is an example of the reproduced RSSM model after 70 episodes of training playing cartpole-swing up from the Deepmind Control suite. The video on the left is the real output and the video on the right is the agents world after VAE decoding.





# Installation
- Follow installation instructions for [Deepmind Control environments]()

# Repo Structure
- **src/Config/**: All config information is stored here including 
    - The structure of the pytorch models
    - Hyperparameters for training
    - Hyperparameters for checkpointing / saving / rollouts
- **src/Models/base.py**: The base transition model for predicting forward in the latent spce
- **src/Models/ssm.py**: The purely stochastic transition model
- **src/Models/rssm.py**: The complete recurrent stochastic transition model
- **src/Models/rnn.py**: The complete recurrent stochastic transition model
- **src/Models/autoencode.py**: The VAE model
- **src/env.py**: Wrappers for gym environments used for testing. Wrappers to use dm_control and openai gym mujoco environments with the same interface
- **src/utils.py**: Utility funcitons
- **src/visualiztion.py**: Files for plotting.


