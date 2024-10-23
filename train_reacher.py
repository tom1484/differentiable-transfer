from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

import time
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp
from diff_trans.envs.wrapped import Reacher_v1

from stable_baselines3 import PPO

num_envs = 64
env = Reacher_v1(num_envs=num_envs)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True)
model.save("reacher_ppo")