import imageio
from tqdm import tqdm
import gymnasium as gym

env = gym.make("Ant-v4", render_mode="rgb_array")
env.reset()

frames = []
for _ in tqdm(range(50)):
    env.step(env.action_space.sample())
    frames.append(env.render())


# save frames to GIF
imageio.mimsave("test_gym_render.gif", frames, fps=env.metadata["render_fps"])
