# PlaNet: Learning Latent Dynamics for Planning from Pixels

This repo contains a pytorch implementation and study of the origiinal Google paper


Planing with known environment dynamics is a highly effective way to solve complex control problems. However, for unobserved environments, we must utilize observations of agent interaction to learn models of the world. 
Deep Planning Network (PlaNet) is an approach that learns the approximate environment dynamics from images, and chooses actions through fast online planning in latent space.
PlaNet uses a latent dynamics model, which contains both deterministic and stochastic transition components, to solve continuous control tasks which exceed the difficulty of tasks previously solved similar methods.
PlaNet is also incredibly data-efficient, and outperforms model-free methods final performance, with on average 200×  fewer environment interactions and similar computation time.




