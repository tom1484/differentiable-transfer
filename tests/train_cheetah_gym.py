from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[1])

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC

num_envs = 1
env = make_vec_env(
    "HalfCheetah-v5",
    n_envs=num_envs,
    env_kwargs={"exclude_current_positions_from_observation": False},
)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)
model.save("half_cheetah_sac")
