from typing import Optional

import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import make

from .base import BaseDiffEnv
from .utils.array import sidx


class DiffHalfCheetah_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                                 | Default   | Min | Max | Joint |
    |-----|-------------------------------------------|-----------|-----|-----|-------|
    | 0   | slide friction of the floor               | 0.4       | 0.2 | 0.6 | slide |
    | 1   | armature inertia of the back thigh rotor  | 0.1       | 0.0 | 0.3 | hinge |
    | 2   | armature inertia of the front thigh rotor | 0.1       | 0.0 | 0.3 | hinge |
    | 3   | damping of the back thigh rotor           | 6.0       | 3.0 | 9.0 | hinge |
    | 4   | damping of the front thigh rotor          | 4.5       | 3.0 | 6.0 | hinge |
    | 5   | mass of the torso                         | 6.2502093 | 4.0 | 8.0 |       |
    """

    def __init__(
        self,
        frame_skip: int,
        reset_noise_scale: float,
        exclude_current_positions_from_observation: bool,
    ):
        observation_dim = 18
        if exclude_current_positions_from_observation:
            observation_dim -= 1

        super().__init__(
            "half_cheetah.xml",
            frame_skip,
            observation_dim,
        )

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # fmt: off
        self.num_parameter = 6
        self.parameter_range = jnp.array(
            [
                [
                    0.2,  # friction
                    0.0, 0.0,  # armature
                    3.0, 3.0,  # damping
                    4.0,  # mass
                ],
                [
                    0.6,  # friction
                    0.3, 0.3,  # armature
                    9.0, 6.0,  # damping
                    8.0,  # mass
                ],
            ]
        )
        # fmt: on

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        pos_noise = random.uniform(
            key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        )
        vel_noise = self._reset_noise_scale * random.normal(key, shape=(self.model.nv,))

        qpos = self.init_qpos + pos_noise
        qvel = self.init_qvel + vel_noise

        return mjx.forward(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def _get_parameter(self) -> jnp.ndarray:
        friction = self.model.geom_friction.copy()
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                friction[0, 0:1],
                armature[sidx(3, 6)],
                damping[sidx(3, 6)],
                mass[1:2],
            ]
        )

    def _set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        friction = self.model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        armature = self.model.dof_armature
        armature = armature.at[sidx(3, 6)].set(parameter[1:3])

        damping = self.model.dof_damping
        damping = damping.at[sidx(3, 6)].set(parameter[3:5])

        mass = self.model.body_mass
        mass = mass.at[1:2].set(parameter[5:6])

        return self.model.replace(
            geom_friction=friction,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _create_gym_env(
        self, parameter: Optional[np.ndarray] = None, **kwargs
    ) -> MujocoEnv:
        gym_env = make("HalfCheetah-v5", **kwargs)

        if parameter is not None:
            model = gym_env.unwrapped.model

            model.geom_friction[0, :1] = parameter[:1]
            model.dof_armature[sidx(3, 6)] = parameter[1:3]
            model.dof_damping[sidx(3, 6)] = parameter[3:5]
            model.body_mass[1:2] = parameter[5:6]

        return gym_env

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        # TODO: Use parallelized version
        if self._exclude_current_positions_from_observation:
            qpos = jnp.concatenate([jnp.zeros(1), states[1:9]])
        else:
            qpos = states[:9]
        qvel = states[9:]
        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        if self._exclude_current_positions_from_observation:
            qpos = env_data.qpos[1:]
        else:
            qpos = env_data.qpos
        return jnp.concatenate([qpos, env_data.qvel])
