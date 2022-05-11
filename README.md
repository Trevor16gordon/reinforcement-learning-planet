# PlaNet: Learning Latent Dynamics for Planning from Pixels

This repo contains a pytorch implementation and study of the origiinal Google paper


Planing with known environment dynamics is a highly effective way to solve complex control problems. However, for unobserved environments, we must utilize observations of agent interaction to learn models of the world. 
Deep Planning Network (PlaNet) is an approach that learns the approximate environment dynamics from images, and chooses actions through fast online planning in latent space.
PlaNet uses a latent dynamics model, which contains both deterministic and stochastic transition components, to solve continuous control tasks which exceed the difficulty of tasks previously solved similar methods.
PlaNet is also incredibly data-efficient, and outperforms model-free methods final performance, with on average 200×  fewer environment interactions and similar computation time.


# Key Results

A video our our agent learning to play cartpole swing up is shown below. This model was trained with the RSSM model for 250 episodes. The video on the left is the real output and the video on the right is the agents world after VAE decoding.

(video)["https://github.com/Trevor16gordon/reinforcement-learning-planet/blob/trevor_develop2/data/results/cartpole_learning.gif"]



# Installation
To install ```pip install -r requirements.txt```
- Follow installation instructions for [Deepmind Control environments](https://github.com/deepmind/dm_control)
- Follow installation instructions for [Mujoco Py environments](https://github.com/openai/mujoco-py)(Optional)


# To Run
```
usage: main.py [-h] [--id ID] [--save-path SAVE_PATH] [--seed S] [--load-model-path LOAD_MODEL_PATH] [--save-results-path SAVE_RESULTS_PATH]
               [--env {Pendulum-v0,MountainCarContinuous-v0,Ant-v2,HalfCheetah-v2,Hopper-v2,Humanoid-v2,HumanoidStandup-v2,InvertedDoublePendulum-v2,InvertedPendulum-v2,Reacher-v2,Swimmer-v2,Walker2d-v2,cartpole-balance,cartpole-swingup,reacher-easy,finger-spin,cheetah-run,ball_in_cup-catch,walker-walk}]
               [--model {ssm,rnn,rssm}] [--render RENDER] [--config CONFIG] [--config-path CONFIG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --id ID               Experiment ID
  --save-path SAVE_PATH
                        Path for saving model check points.
  --seed S              Random seed
  --load-model-path LOAD_MODEL_PATH
                        If given, model will start from this path
  --save-results-path SAVE_RESULTS_PATH
                        Path to output file
  --env {Pendulum-v0,MountainCarContinuous-v0,Ant-v2,HalfCheetah-v2,Hopper-v2,Humanoid-v2,HumanoidStandup-v2,InvertedDoublePendulum-v2,InvertedPendulum-v2,Reacher-v2,Swimmer-v2,Walker2d-v2,cartpole-balance,cartpole-swingup,reacher-easy,finger-spin,cheetah-run,ball_in_cup-catch,walker-walk}
                        Gym/Control Suite environment
  --model {ssm,rnn,rssm}
                        Select the State Space Model to Train.
  --render RENDER       Render environment
  --config CONFIG       Specify the yaml file to use in setting up experiment.
  --config-path CONFIG_PATH
                        Specify the directory the config file lives in.
```


# Repo Structure
- **src/main.py**: The entry point for training any of the models

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


