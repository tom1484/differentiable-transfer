from typing import List

from diff_trans.envs.base import BaseDiffEnv
from diff_trans.envs.gym import BaseEnv


def generate_trajectory_pair(env0: BaseEnv, env1: BaseEnv, length: int):
    obs0 = env0.reset()
    obs1 = env1.reset()

    obs_list0 = [obs0]
    obs_list1 = [obs1]

    for _ in range(length):
        env0.step(env0.action_space.sample())
        env1.step(env1.action_space.sample())

    # return env0, env1

def generate_dataset(envs: List[BaseEnv]):
    for env in envs:
        env.reset()
        