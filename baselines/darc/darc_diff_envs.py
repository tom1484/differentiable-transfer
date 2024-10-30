# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments used in the DARC experiments."""
import tempfile
import gin
import gym
from gym import utils
from gym.envs.mujoco import ant
from gym.envs.mujoco import half_cheetah
from gym.envs.mujoco import hopper
from gym.envs.mujoco import humanoid
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import walker2d
import gym.spaces
import numpy as np
import reacher_7dof
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from diff_trans.envs.gym import ant_v5
from diff_trans.envs.gym import half_cheetah_v5
from diff_trans.envs.gym import humanoid_v5
from diff_trans.envs.gym import walker2d_v5
from diff_trans.envs.gym import reacher_v5


class BrokenJoint(gym.Wrapper):
  """Wrapper that disables one coordinate of the action, setting it to 0."""

  def __init__(self, env, broken_joint):
    super(BrokenJoint, self).__init__(env)
    # Change dtype of observation to be float32
    self.observation_space = gym.spaces.Box(
        low=self.observation_space.low,
        high=self.observation_space.high,
        dtype=np.float32,
    )
    if broken_joint is not None:
      assert 0 <= broken_joint <= len(self.action_space.low) - 1
    self.broken_joint = broken_joint

  def step(self, action):
    action = action.copy()
    if self.broken_joint is not None:
      action[self.broken_joint] = 0
    return super(BrokenJoint, self).step(action)


@gin.configurable
def get_broken_joint_env(mode, env_name, broken_joint=0):
  """Creates an environment with a broken joint."""
  if env_name == "ant":
    env = ant_v5.Ant_v5()
  elif env_name == "half_cheetah":
    env = half_cheetah_v5.HalfCheetah_v5()
  elif env_name == "reacher":
    env = reacher_v5.Reacher_v5()
  else:
    raise NotImplementedError
  if mode == "sim":
    env = BrokenJoint(env, broken_joint=None)
  else:
    assert mode == "real"
    env = BrokenJoint(env, broken_joint=broken_joint)
  env = suite_gym.wrap_env(env, max_episode_steps=1000)
  return tf_py_environment.TFPyEnvironment(env)


class FallingEnv(gym.Wrapper):
  """Wrapper that disables the termination condition."""

  def __init__(self, env, ignore_termination=False):
    self._ignore_termination = ignore_termination
    super(FallingEnv, self).__init__(env)

  def step(self, a):
    ns, r, done, info = super(FallingEnv, self).step(a)
    r = 0.0
    if self._ignore_termination:
      done = False
    return ns, r, done, info


def get_falling_env(mode, env_name):
  """Creates an environment with the termination condition disabled."""
  if env_name == "hopper":
    env = hopper.HopperEnv()
  elif env_name == "humanoid":
    env = humanoid.HumanoidEnv()
  elif env_name == "walker":
    env = walker2d.Walker2dEnv()
  else:
    raise NotImplementedError
  if mode == "sim":
    env = FallingEnv(env, ignore_termination=True)
  else:
    assert mode == "real"
    env = FallingEnv(env, ignore_termination=False)

  env = suite_gym.wrap_env(env, max_episode_steps=300)
  return tf_py_environment.TFPyEnvironment(env)
