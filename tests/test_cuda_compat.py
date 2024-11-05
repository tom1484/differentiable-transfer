from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[0])

from tqdm import tqdm

import torch
from torch.optim import Adam
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from diff_trans.envs.gym_wrapper import get_env

from baselines.darc.agent import DarcAgent, evaluate_policy
from baselines.darc.models import ActorNet, QValueNet, ValueNet, ClassifierNet
from baselines.darc.agent import SimpleCollector

num_envs = 10
learning_rate = 1e-3

Env = get_env("InvertedPendulum-v5")
env = Env(num_envs=num_envs)
eval_env = Env(num_envs=100)