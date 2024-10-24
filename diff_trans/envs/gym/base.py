from pprint import pformat
import time
from typing import Any, Dict, Optional, Tuple, List, Type

import numpy as np

import gymnasium as gym

from gymnasium.spaces import Box

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import env_util

import jax
from jax import numpy as jnp
from mujoco import mjx

from ... import envs


DEFAULT_SIZE = 480


class BaseEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        diff_env: envs.BaseDiffEnv,
        max_episode_steps: int,
        observation_space: Box,
        action_space: Box,
    ) -> None:
        self.diff_env = diff_env
        self.num_env = num_envs
        self.num_envs = num_envs

        self.max_episode_steps = max_episode_steps

        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        obs = self._reset_all()

        self.steps = np.zeros(self.num_env, dtype=int)
        self.rewards = np.zeros((self.num_env, 1), dtype=np.float32)
        self.time_start = np.array([time.time() for _ in range(self.num_env)])

        return obs

    def _reset_all(self) -> np.ndarray:
        rng = jax.random.PRNGKey(time.time_ns())
        rng = jax.random.split(rng, self.num_envs)

        self._states = self.diff_env.reset_vj(rng)
        obs = self.diff_env._get_obs_vj(self._states)

        return np.asarray(obs).copy()

    def _reset_at(self, at: np.ndarray) -> np.ndarray:
        n = at.astype(np.int32).sum()

        rng = jax.random.PRNGKey(time.time_ns())
        rng = jax.random.split(rng, n)

        data = self._states
        new_data = self.diff_env.reset_vj(rng)
        obs = self.diff_env._get_obs_vj(new_data)

        new_qpos = data.qpos.at[at].set(new_data.qpos)
        new_qvel = data.qvel.at[at].set(new_data.qvel)

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
        pass

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

    def get_state_vector(self, data: mjx.Data) -> jnp.ndarray:
        return self.diff_env._get_state_vector_batch(data)

    def reshape_info(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            dict([item[0], item[1][e]] for item in info.items())
            for e in range(self.num_envs)
        ]
