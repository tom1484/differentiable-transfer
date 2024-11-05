from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[1])

from diff_trans.envs.gym_wrapper import Reacher_v5
from stable_baselines3 import SAC

num_envs = 1
env = Reacher_v5(num_envs=num_envs)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True)
model.save("reacher_sac")
