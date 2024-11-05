from typing import Tuple, Optional

from jax import numpy as jnp
import gymnasium as gym
import numpy as np
import random

from diff_trans.envs.gym_wrapper import BaseEnv
from diff_trans.envs.gym_wrapper.utils.entry import get_env

# from mujoco._structs import MjModel


# def get_geom_id(model: MjModel, geom_name: str):
#     names = [i.decode() for i in model.names.split(b"\x00")][1:]
#     for i, name in enumerate(names):
#         if geom_name == name:
#             for index, body_id in enumerate(model.geom_bodyid):
#                 if i == body_id:
#                     return index

#     raise ValueError(f"Geometry name '{geom_name}' not found in the model.")


# hardcoded for fetch_slide_2, with only friction
# class RandomizedEnvironment:
#     """Randomized environment class"""

#     def __init__(self, experiment, parameter_ranges, goal_range):
#         self._experiment = experiment
#         self._parameter_ranges = parameter_ranges
#         self._goal_range = goal_range
#         self._params = [0]
#         random.seed(123)

#     def sample_env(self, render_mode=None, renderer=None):
#         mini = self._parameter_ranges[0]
#         maxi = self._parameter_ranges[1]
#         pick = mini + (maxi - mini) * random.random()

#         self._params = np.array([pick])
#         self._env = gym.make(self._experiment, render_mode=render_mode, renderer=renderer)
#         self._env.env.reward_type = "dense"
#         self._env.env.model.geom_friction[get_geom_id(self._env.env.model, "object0")] = [pick, 0.005, 0.0001]

#     def get_env(self):
#         """
#         Returns a randomized environment and the vector of the parameter
#         space that corresponds to this very instance
#         """
#         return self._env, self._params

#     def close_env(self):
#         self._env.close()

#     def get_goal(self):
#         return


class RandomizedEnvironment:
    """Randomized environment class"""

    def __init__(self, env_name: str, parameter_mask: Optional[np.ndarray] = None, **kwargs):
        ENV = get_env(env_name)

        self.gym_env = ENV(**kwargs)
        self.env = self.gym_env.diff_env

        default_parameter = self.env.get_parameter()
        if parameter_mask is None:
            parameter_mask = np.zeros(default_parameter.shape[0], dtype=bool)
        else:
            assert parameter_mask.shape[0] == default_parameter.shape[0]

        self.default_parameter = default_parameter
        self.parameter_mask = jnp.array(parameter_mask)
        self.parameter_range = self.env.parameter_range

    def sample_env(self):
        mini = self.parameter_range[0]
        maxi = self.parameter_range[1]
        value = mini + (maxi - mini) * random.random()

        # keep the default values for the parameters that are not being randomized
        value = value.at[self.parameter_mask].set(
            self.default_parameter[self.parameter_mask]
        )
        self.env.model = self.env.set_parameter(value)

    def get_env(self) -> Tuple[BaseEnv, jnp.ndarray]:
        return self.gym_env, self.env.get_parameter()