from typing import Optional

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from mujoco import mjx
from gymnasium import Env

from .base import BaseDiffEnv


class DiffInvertedPendulum_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                         | Default   | Min | Max  | Joint |
    |-----|-----------------------------------|-----------|-------|------|-------|
    | 0   | friction loss of the slider       |  0.0      | 0.001 | None | slide |
    | 1   | friction loss of the pole rotor   |  0.0      | 0.001 | None | hinge |
    | 2   | armature mass of the cart         |  0.0      | 0.001 | None | slide |
    | 3   | armature inertia of the pole      |  0.0      | 0.001 | None | hinge |
    | 4   | damping of the slider             |  1.0      | 0.001 | None | slide |
    | 5   | damping of the pole rotor         |  1.0      | 0.001 | None | hinge |
    | 6   | mass of the cart                  | 10.471975 | 0.001 | None |       |
    | 7   | mass of the joint                 |  5.018591 | 0.001 | None |       |
    """

    def __init__(
        self,
        frame_skip: int,
        reset_noise_scale: float,
    ):
        observation_dim = 4
        super().__init__("inverted_pendulum.xml", frame_skip, observation_dim)

        self._reset_noise_scale = reset_noise_scale

        # fmt: off
        self.num_parameter = 8
        self.parameter_range = jnp.array(
            [
                [
                    0.001, 0.001,  # friction loss
                    0.001, 0.001,  # armature
                    0.001, 0.001,  # damping
                    0.001, 0.001,  # mass
                ],
                [
                    jnp.inf, jnp.inf,  # friction loss
                    jnp.inf, jnp.inf,  # armature
                    jnp.inf, jnp.inf,  # damping
                    jnp.inf, jnp.inf,  # mass
                ],
            ]
        )
        # fmt: on

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        noise = random.uniform(
            key,
            shape=(self.model.nq + self.model.nv,),
            minval=noise_low,
            maxval=noise_high,
        )

        qpos = self.init_qpos + noise[:2]
        qvel = self.init_qvel + noise[2:]

        return mjx.forward(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def _get_parameter(self) -> jax.Array:
        frictionloss = self.model.dof_frictionloss.copy()
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                frictionloss[np.array([0, 1])],
                armature[np.array([0, 1])],
                damping[np.array([0, 1])],
                mass[np.array([1, 2])],
            ]
        )

    def _set_parameter(self, parameter: jax.Array) -> mjx.Model:
        frictionloss = self.model.dof_frictionloss
        frictionloss = frictionloss.at[np.array([0, 1])].set(parameter[:2])

        armature = self.model.dof_armature
        armature = armature.at[np.array([0, 1])].set(parameter[2:4])

        damping = self.model.dof_damping
        damping = damping.at[np.array([0, 1])].set(parameter[4:6])

        mass = self.model.body_mass
        mass = mass.at[np.array([1, 2])].set(parameter[6:8])

        return self.model.replace(
            dof_frictionloss=frictionloss,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _create_gym_env(self, parameter: Optional[np.ndarray] = None) -> Env:
        raise NotImplementedError()

    def _state_to_data(self, data: mjx.Data, states: jax.Array) -> mjx.Data:
        # TODO: Use parallelized version
        qpos = states[:2]
        qvel = states[2:]
        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jax.Array) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        return jnp.concatenate([env_data.qpos, env_data.qvel])
