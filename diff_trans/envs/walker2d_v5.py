from typing import Optional

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from mujoco import mjx
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import make

from .base import BaseDiffEnv


class DiffWalker2d_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                           | Default   | Min   | Max  | Joint |
    |-----|-------------------------------------|-----------|-------|------|-------|
    | 0   | slide friction of the floor         | 0.7       | 0.001 | None | slide |
    | 1   | armature inertia of the right thigh | 0.01      | 0.001 | None | hinge |
    | 2   | armature inertia of the right leg   | 0.01      | 0.001 | None | hinge |
    | 3   | armature inertia of the right foot  | 0.01      | 0.001 | None | hinge |
    | 4   | armature inertia of the left thigh  | 0.01      | 0.001 | None | hinge |
    | 5   | armature inertia of the left leg    | 0.01      | 0.001 | None | hinge |
    | 6   | armature inertia of the left foot   | 0.01      | 0.001 | None | hinge |
    | 7   | damping of the right thigh          | 0.1       | 0.001 | None | hinge |
    | 8   | damping of the right leg            | 0.1       | 0.001 | None | hinge |
    | 9   | damping of the right foot           | 0.1       | 0.001 | None | hinge |
    | 10  | damping of the left thigh           | 0.1       | 0.001 | None | hinge |
    | 11  | damping of the left leg             | 0.1       | 0.001 | None | hinge |
    | 12  | damping of the left foot            | 0.1       | 0.001 | None | hinge |
    | 13  | mass of the torso                   | 3.6651914 | 0.001 | None |       |
    | 14  | mass of the right thigh             | 4.0578904 | 0.001 | None |       |
    | 15  | mass of the right leg               | 2.7813568 | 0.001 | None |       |
    | 16  | mass of the right foot              | 3.1667254 | 0.001 | None |       |
    | 17  | mass of the left thigh              | 4.0578904 | 0.001 | None |       |
    | 18  | mass of the left leg                | 2.7813568 | 0.001 | None |       |
    | 19  | mass of the left foot               | 3.1667254 | 0.001 | None |       |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
    ):
        observation_dim = 18
        if exclude_current_positions_from_observation:
            observation_dim -= 1

        super().__init__(
            "walker2d.xml",
            frame_skip,
            observation_dim,
        )

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # fmt: off
        self.num_parameter = 14
        self.parameter_range = jnp.array(
            [
                [
                    0.001,  # friction
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  # armature
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  # damping
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  # mass
                ],
                [
                    jnp.inf,  # friction
                    jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,  # armature
                    jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,  # damping
                    jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,  # mass
                ],
            ]
        )
        # fmt: on

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + random.uniform(
            key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        )
        qvel = self.init_qvel + random.uniform(
            key, shape=(self.model.nv,), minval=noise_low, maxval=noise_high
        )

        return mjx.step(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def _get_parameter(self) -> jax.Array:
        friction = self.model.geom_friction.copy()
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                friction[0, 0:1],
                armature[3:9],
                damping[3:9],
                mass[1:2],
            ]
        )

    def _set_parameter(self, parameter: jax.Array) -> mjx.Model:
        friction = self.model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        armature = self.model.dof_armature
        armature = armature.at[3:9].set(parameter[1:7])

        damping = self.model.dof_damping
        damping = damping.at[3:9].set(parameter[7:13])

        mass = self.model.body_mass
        mass = mass.at[1:2].set(parameter[13:14])

        return self.model.replace(
            geom_friction=friction,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _create_gym_env(self, parameter: Optional[np.ndarray] = None, **kwargs) -> MujocoEnv:
        gym_env = make("Walker2d-v5", **kwargs)

        if parameter is not None:
            model = gym_env.unwrapped.model

            model.geom_friction[0, :1] = parameter[:1]
            model.dof_armature[3:9] = parameter[1:7]
            model.dof_damping[3:9] = parameter[7:13]
            model.body_mass[1:2] = parameter[13:14]

        return gym_env

    def _state_to_data(self, data: mjx.Data, states: jax.Array) -> mjx.Data:
        if self._exclude_current_positions_from_observation:
            states = jnp.concatenate([jnp.zeros(1), states])

        qpos = states[:9]
        qvel = states[9:]

        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jax.Array) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        qpos = data.qpos
        qvel = jnp.clip(data.qvel, -10, 10)

        if self._exclude_current_positions_from_observation:
            qpos = qpos[1:]

        return jnp.concatenate([qpos, qvel])
