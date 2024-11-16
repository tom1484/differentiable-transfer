import time
from typing import Any, Dict, Optional, Tuple, List, Type, Union

import numpy as np
import mujoco

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium import Env

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import env_util

import jax
from jax import numpy as jnp
from mujoco import mjx

from ..base import BaseDiffEnv


DEFAULT_SIZE = 480


class BaseEnv(VecEnv):
    def __init__(
        self,
        diff_env: BaseDiffEnv,
        num_envs: int,
        max_episode_steps: int,
        observation_space: Box,
        action_space: Box,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        precompile: bool = True,
    ) -> None:
        """
        Initialize the environment.
        """
        self.num_env = num_envs
        self.num_envs = num_envs

        self.max_episode_steps = max_episode_steps

        self.diff_env = diff_env

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_parameter = diff_env.num_parameter

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / diff_env.dt))}, Actual value: {self.metadata["render_fps"]}'

        self._init_renderer(
            diff_env.mj_model,
            diff_env.mj_data,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
            default_camera_config,
        )

        # Compile the JIT functions for the environment for faster runtime
        if precompile:
            self.diff_env.compile(num_envs)

    def reset(self):
        """
        Reset the environment.
        """
        obs = self._reset_all()

        self.steps = np.zeros(self.num_env, dtype=int)
        self.rewards = np.zeros((self.num_env, 1), dtype=np.float32)
        self.time_start = np.array([time.time() for _ in range(self.num_env)])

        return obs

    def _reset_all(self) -> np.ndarray:
        """
        Reset the underlying differentiable environment.
        """
        rng = jax.random.PRNGKey(time.time_ns())
        rng = jax.random.split(rng, self.num_envs)

        self._states = self.diff_env.reset_vj(rng)
        obs = self.diff_env._get_obs_vj(self._states)

        return np.asarray(obs).copy()

    def _reset_at(self, at: np.ndarray) -> np.ndarray:
        """
        Reset the underlying differentiable environment at the given indices.
        """
        rng = jax.random.PRNGKey(time.time_ns())
        rng = jax.random.split(rng, self.num_envs)

        data = self._states
        new_data = self.diff_env.reset_vj(rng)
        obs = self.diff_env._get_obs_vj(new_data)[at]

        new_qpos = data.qpos.at[at].set(new_data.qpos[at])
        new_qvel = data.qvel.at[at].set(new_data.qvel[at])

        data = data.replace(qpos=new_qpos, qvel=new_qvel)
        self._states = data

        return np.asarray(obs).copy()

    def _update_steps(
        self,
        observation: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Determine if the episode is done and update the steps and rewards.
        """
        if self.steps.max() >= self.rewards.shape[1]:
            self.rewards = np.hstack([self.rewards, self.rewards])

        self.rewards[:, self.steps] = reward.flatten()
        self.steps += 1

        # Truncate
        done[self.steps >= self.max_episode_steps] = True

        # Calculate episode info for done environments
        if info is None:
            info = [{} for _ in range(self.num_env)]

        needs_reset = np.zeros(self.num_env, dtype=bool)
        for idx, d in enumerate(done):
            if not d:
                continue

            steps = self.steps[idx]
            r = self.rewards[idx][:steps]
            t = time.time() - self.time_start[idx]

            episode = {}
            episode["r"] = np.sum(r)
            episode["l"] = steps
            episode["t"] = t

            info[idx]["episode"] = episode

            needs_reset[idx] = True

        # Reset done environments
        reset_num = np.sum(needs_reset.astype(np.int32))
        if reset_num > 0:
            reset_obs = self._reset_at(needs_reset)
            observation[needs_reset] = reset_obs
            self.steps[needs_reset] = 0

        return observation, done, info

    def _step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        NotImplementedError()

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step the environment.
        """
        observation, reward, done, info = self._step_wait()
        observation = np.asarray(observation).copy()
        reward = np.asarray(reward).copy()
        done = np.asarray(done).copy()
        # Update the done flag and auto reset the environments if needed
        observation, done, info = self._update_steps(
            observation, reward, done, info=info
        )

        return observation, reward, done, info

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = jnp.array(actions)

    def close(self) -> None:
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def _get_indices_len(self, indices) -> int:
        if indices is None:
            return self.num_env
        if isinstance(indices, int):
            return 1
        return len(indices)

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        num = self._get_indices_len(indices)
        attr = getattr(self, attr_name)
        return [attr for _ in range(num)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        val = value if indices is None else value[0]
        setattr(self, attr_name, val)

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> List[Any]:
        num = self._get_indices_len(indices)

        method = getattr(self, method_name)
        value = method(*method_args, **method_kwargs)
        return [value for _ in range(num)]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices=None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        num = self._get_indices_len(indices)
        result = env_util.is_wrapped(self, wrapper_class)
        return [result for _ in range(num)]

    def get_state_vector(self, data: mjx.Data) -> jax.Array:
        return self.diff_env._get_state_vector_batch(data)

    def reshape_info(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            dict([item[0], item[1][e]] for item in info.items())
            for e in range(self.num_envs)
        ]

    def get_model_parameter(self) -> jax.Array:
        return self.diff_env._get_parameter()

    def set_model_parameter(self, parameter: jax.Array):
        self.diff_env.model = self.diff_env._set_parameter(parameter)

    def create_gym_env(
        self, parameter: Optional[np.ndarray] = None, **kwargs
    ) -> Env:
        return self.diff_env._create_gym_env(parameter, **kwargs)

    def update_gym_env(self, gym_env: Env, parameter: jax.Array):
        self.diff_env._update_gym_env(gym_env, parameter)

    def _init_renderer(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        render_mode: Optional[str],
        width: int,
        height: int,
        camera_id: Optional[int],
        camera_name: Optional[str],
        default_camera_config: Optional[Dict[str, Union[float, int]]],
    ) -> None:
        """
        Initialize the renderer.
        """
        # Rendering settings
        self.width = width
        self.height = height

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        # model.vis.global_.offwidth = self.width
        # model.vis.global_.offheight = self.height
        self.mujoco_renderer = MujocoRenderer(
            model,
            data,
            default_cam_config=default_camera_config,
            width=self.width,
            height=self.height,
        )

    def _render_single(self, select_env: int) -> np.ndarray:
        """
        Render a single environment.
        """
        renderer_model = self.mujoco_renderer.model
        renderer_data = self.mujoco_renderer.data

        # Update kinematics
        renderer_data.qpos = np.array(self._states.qpos[select_env])
        mujoco.mj_forward(renderer_model, renderer_data)

        return self.mujoco_renderer.render(
            self.render_mode,
            # self.camera_id,
            # self.camera_name,
        )

    def render(self, select_envs: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        Render the environment for the given indices.
        """
        # Needs fixing due to upgrading to gymnasium 1.0.0
        # raise NotImplementedError()

        if isinstance(select_envs, int):
            return self._render_single(select_envs)

        if select_envs is None:
            select_envs = range(self.num_env)

        frames = [self._render_single(e) for e in select_envs]
        return np.stack(frames)
