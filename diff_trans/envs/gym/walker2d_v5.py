from typing import Tuple, Union

import numpy as np

from gymnasium.spaces import Box

from jax import numpy as jnp
from mujoco import mjx

from .base import BaseEnv
from ... import envs, sim


class Walker2d_v5(BaseEnv):

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
    ):
        env = envs.DiffWalker2d_v5()

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.state_dim,),
            dtype=np.float32,
        )
        action_space = Box(
            low=env.control_range[0], high=env.control_range[1], dtype=np.float32
        )

        super().__init__(
            num_envs, env, max_episode_steps, observation_space, action_space
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def healthy_reward(self, data: mjx.Data) -> jnp.ndarray:
        return self.is_healthy(data) * self._healthy_reward

    def control_cost(self, control: jnp.ndarray) -> jnp.ndarray:
        control_cost = self._ctrl_cost_weight * jnp.sum(control**2, axis=1)
        return control_cost

    def is_healthy(self, data: mjx.Data) -> jnp.ndarray:
        qpos = data.qpos
        z, angle = qpos[:, 1], qpos[:, 2]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = jnp.logical_and(min_z <= z, z <= max_z)
        healthy_angle = jnp.logical_and(min_angle <= angle, angle <= max_angle)
        is_healthy = jnp.logical_and(healthy_z, healthy_angle)

        return is_healthy

    def _get_reward(
        self, data: mjx.Data, x_velocity: jnp.ndarray, control: jnp.ndarray
    ) -> jnp.ndarray:
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward(data)
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(control)
        costs = ctrl_cost

        reward = rewards - costs
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _step_wait(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        data = self._states
        control = self._actions

        x_pos_before = data.qpos[:, 0]
        data = sim.step_vj(self.diff_env, self.diff_env.model, data, control)
        self._states = data

        x_pos_after = data.qpos[:, 0]
        x_velocity = (x_pos_after - x_pos_before) / self.diff_env.dt

        observation = self.diff_env._get_obs_vj(data)
        reward, reward_info = self._get_reward(data, x_velocity, control)

        info = self.reshape_info(
            {
                "x_position": x_pos_after,
                "z_distance_from_origin": data.qpos[:, 1] - self.diff_env.init_qpos[1],
                "x_velocity": x_velocity,
                **reward_info,
            }
        )

        terminated = jnp.logical_and(
            jnp.logical_not(self.is_healthy(data)), self._terminate_when_unhealthy
        )

        return observation, reward, terminated, info
