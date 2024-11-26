from typing import Optional

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from mujoco import mjx
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import make

from .base import BaseDiffEnv


class DiffAnt_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                      | Default   | Min   | Max  | Joint |
    |-----|--------------------------------|-----------|-------|------|-------|
    | 0   | slide friction of the floor    | 1.0       | 0.001 | None | slide |
    | 1   | armature inertia of the hip1   | 1.0       | 0.001 | None | hinge |
    | 2   | armature inertia of the ankle1 | 1.0       | 0.001 | None | hinge |
    | 3   | armature inertia of the hip2   | 1.0       | 0.001 | None | hinge |
    | 4   | armature inertia of the ankle2 | 1.0       | 0.001 | None | hinge |
    | 5   | armature inertia of the hip3   | 1.0       | 0.001 | None | hinge |
    | 6   | armature inertia of the ankle3 | 1.0       | 0.001 | None | hinge |
    | 7   | armature inertia of the hip4   | 1.0       | 0.001 | None | hinge |
    | 8   | armature inertia of the ankle4 | 1.0       | 0.001 | None | hinge |
    | 9   | damping of the hip1            | 1.0       | 0.001 | None | hinge |
    | 10  | damping of the ankle1          | 1.0       | 0.001 | None | hinge |
    | 11  | damping of the hip2            | 1.0       | 0.001 | None | hinge |
    | 12  | damping of the ankle2          | 1.0       | 0.001 | None | hinge |
    | 13  | damping of the hip3            | 1.0       | 0.001 | None | hinge |
    | 14  | damping of the ankle3          | 1.0       | 0.001 | None | hinge |
    | 15  | damping of the hip4            | 1.0       | 0.001 | None | hinge |
    | 16  | damping of the ankle4          | 1.0       | 0.001 | None | hinge |
    | 17  | mass of the torso              | 0.3272492 | 0.001 | None |       |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
    ):
        observation_dim = 29
        observation_dim -= 2 * exclude_current_positions_from_observation
        # TODO: Add contact force observation

        super().__init__(
            "ant.xml",
            frame_skip,
            observation_dim,
        )

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        # fmt: off
        self.num_parameter = 18
        self.parameter_range = jnp.array(
            [
                [
                    0.001,  # friction
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  # armature
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  # damping
                    0.001,  # mass
                ],
                [
                    jnp.inf,  # friction
                    jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,  # armature
                    jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,  # damping
                    jnp.inf,  # mass
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
        qvel = self.init_qvel + self._reset_noise_scale * random.normal(
            key, shape=(self.model.nv,)
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
                armature[6:14],
                damping[6:14],
                mass[1:2],
            ]
        )

    def _set_parameter(self, parameter: jax.Array) -> mjx.Model:
        friction = self.model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        armature = self.model.dof_armature
        armature = armature.at[6:14].set(parameter[1:9])

        damping = self.model.dof_damping
        damping = damping.at[6:14].set(parameter[9:17])

        mass = self.model.body_mass
        mass = mass.at[1:2].set(parameter[17:18])

        return self.model.replace(
            geom_friction=friction,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _create_gym_env(
        self, parameter: Optional[np.ndarray] = None, **kwargs
    ) -> MujocoEnv:
        gym_env = make("Ant-v5", **kwargs)

        if parameter is not None:
            model = gym_env.unwrapped.model

            model.geom_friction[0, :1] = parameter[:1]
            model.dof_armature[6:14] = parameter[1:9]
            model.dof_damping[6:14] = parameter[9:17]
            model.body_mass[1:2] = parameter[17:18]

        return gym_env

    def _state_to_data(self, data: mjx.Data, states: jax.Array) -> mjx.Data:
        if self._exclude_current_positions_from_observation:
            states = jnp.concatenate([jnp.zeros(2), states])

        qpos = states[:15]
        qvel = states[15:29]

        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jax.Array) -> mjx.Data:
        return data.replace(ctrl=control)

    # TODO: Add contact force observation
    # def contact_forces(self, data: mjx.Data) -> jax.Array:
    #     raw_contact_forces = data.cfrc_ext.flatten()
    #     min_value, max_value = self._contact_force_range
    #     contact_forces = jnp.clip(raw_contact_forces, min_value, max_value)

    #     return contact_forces

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        qpos = data.qpos
        qvel = data.qvel

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        # TODO: Add contact force observation
        # if self._include_cfrc_ext_in_observation:
        #     contact_force = self.contact_forces(data)[1:]
        #     return jnp.concatenate([qpos, qvel, contact_force])

        return jnp.concatenate([qpos, qvel])
