# SigCuriousRLWorld
SigCuriousRLWorld is research experiment code exploring a merge between JEPA-based world models and reinforcement learning from raw image observations.
It uses a LeWorldModel-style latent predictor with SIGReg to keep representations stable and avoid collapse.
The agent learns with PPO, using world-model prediction error as intrinsic reward and latent features for policy learning.

