import gymnasium as gym
import numpy as np
import random

from mujoco._structs import MjModel


def get_geom_id(model: MjModel, geom_name: str):
    names = [i.decode() for i in model.names.split(b"\x00")][1:]
    for i, name in enumerate(names):
        if geom_name == name:
            for index, body_id in enumerate(model.geom_bodyid):
                if i == body_id:
                    return index

    raise ValueError(f"Geometry name '{geom_name}' not found in the model.")



# hardcoded for fetch_slide_2, with only friction
class RandomizedEnvironment:
    """Randomized environment class"""

    def __init__(self, experiment, parameter_ranges, goal_range):
        self._experiment = experiment
        self._parameter_ranges = parameter_ranges
        self._goal_range = goal_range
        self._params = [0]
        random.seed(123)

    def sample_env(self, render_mode=None, renderer=None):
        mini = self._parameter_ranges[0]
        maxi = self._parameter_ranges[1]
        pick = mini + (maxi - mini) * random.random()

        self._params = np.array([pick])
        self._env = gym.make(self._experiment, render_mode=render_mode, renderer=renderer)
        self._env.env.reward_type = "dense"
        self._env.env.model.geom_friction[get_geom_id(self._env.env.model, "object0")] = [pick, 0.005, 0.0001]

    def get_env(self):
        """
        Returns a randomized environment and the vector of the parameter
        space that corresponds to this very instance
        """
        return self._env, self._params

    def close_env(self):
        self._env.close()

    def get_goal(self):
        return
