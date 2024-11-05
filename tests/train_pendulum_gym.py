from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC

num_envs = 4
env = make_vec_env("InvertedPendulum-v4", n_envs=num_envs)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True)
model.save("inverted_pendulum_sac")