from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

import imageio
from tqdm import tqdm
from diff_trans.envs.gym_wrapper import get_env

Env = get_env("Reacher-v5")
env = Env(num_envs=1, render_mode="rgb_array")
env.reset()

frames = []
for _ in tqdm(range(250)):
    env.step(env.action_space.sample()[None, :])
    frames.append(env.render(0))


# save frames to GIF
imageio.mimsave("test_render.gif", frames, fps=env.metadata["render_fps"])
