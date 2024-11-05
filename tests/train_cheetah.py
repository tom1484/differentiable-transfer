from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[1])

from diff_trans.envs.gym_wrapper import HalfCheetah_v5
from stable_baselines3 import SAC

num_envs = 4
env = HalfCheetah_v5(num_envs=num_envs)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)
model.save("half_cheetah_sac")
