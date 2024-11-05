from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

import imageio
from tqdm import tqdm
from diff_trans.envs.gym_wrapper import get_env

from stable_baselines3 import SAC
import sys

ckpt_path = sys.argv[1]

Env = get_env("Reacher-v5")
env = Env(num_envs=1, render_mode="rgb_array")
obs = env.reset()

model = SAC("MlpPolicy", env)
model.load(ckpt_path)

frames = [env.render(0)]
for _ in tqdm(range(250)):
    action, _ = model.predict(obs)
    obs, _, _, _ = env.step(action)
    frames.append(env.render(0))

# save frames to GIF
imageio.mimsave("render_reacher.gif", frames, fps=env.metadata["render_fps"])
