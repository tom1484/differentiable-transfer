from experiments.env import set_env_vars

set_env_vars()

import numpy as np
from jax import numpy as jnp
from stable_baselines3 import PPO, DDPG

# from stable_baselines3.common.evaluation import evaluate_policy
from diff_trans.envs.gym import get_env, BaseEnv
from diff_trans.utils.rollout import evaluate_policy
from diff_trans import sim

from diff_trans.utils.callbacks import EvalCallback

# from stable_baselines3.common.callbacks import EvalCallback

num_envs = 128
timesteps = 2000000
eval_timesteps = 100000

env_type = get_env("HalfCheetah-v5")
env = env_type(num_envs=num_envs, max_episode_steps=1000)
eval_env = env_type(num_envs=1024, max_episode_steps=1000)


def log_wandb(metrics):
    print(metrics)


callback = EvalCallback(
    eval_env=eval_env,
    n_eval_episodes=num_envs,
    eval_freq=eval_timesteps // num_envs,
    callback_on_log=log_wandb,
    verbose=1,
)

# model = DDPG("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
# model = model.learn(total_timesteps=200000, progress_bar=True, callback=callback)
model = model.learn(total_timesteps=timesteps, callback=callback)
# model = model.save("ppo_half_cheetah")
# model = model.load("ppo_half_cheetah")
# model = model.load("./models/parameter_convergence/single/half_cheetah/ppo_inverted_pendulum- 2-00.zip")
model.save("test_PPO_half_cheetah")

# model.save("test_PPO_half_cheetah")
mean_return, std_return = evaluate_policy(
    eval_env, model, n_eval_episodes=1000, return_episode_rewards=False
)
print(mean_return, std_return)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
