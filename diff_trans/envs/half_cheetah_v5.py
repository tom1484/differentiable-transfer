from typing import Optional

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from mujoco import mjx
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import make

from .base import BaseDiffEnv


class DiffHalfCheetah_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                                 | Default   | Min   | Max   | Joint |
    |-----|-------------------------------------------|-----------|-------|-------|-------|
    | 0   | slide friction of the floor               | 0.4       | 0.001 | None  | slide |
    | 1   | armature inertia of the back thigh rotor  | 0.1       | 0.001 | None  | hinge |
    | 2   | armature inertia of the back shin rotor   | 0.1       | 0.001 | None  | hinge |
    | 3   | armature inertia of the back foot rotor   | 0.1       | 0.001 | None  | hinge |
    | 4   | armature inertia of the front thigh rotor | 0.1       | 0.001 | None  | hinge |
    | 5   | armature inertia of the front shin rotor  | 0.1       | 0.001 | None  | hinge |
    | 6   | armature inertia of the front foot rotor  | 0.1       | 0.001 | None  | hinge |
    | 7   | damping of the back thigh rotor           | 6.0       | 0.001 | None  | hinge |
    | 8   | damping of the back shin rotor            | 4.5       | 0.001 | None  | hinge |
    | 9   | damping of the back foot rotor            | 3.0       | 0.001 | None  | hinge |
    | 10  | damping of the front thigh rotor          | 4.5       | 0.001 | None  | hinge |
    | 11  | damping of the front shin rotor           | 3.0       | 0.001 | None  | hinge |
    | 12  | damping of the front foot rotor           | 1.5       | 0.001 | None  | hinge |
    | 13  | mass of the torso                         | 6.2502093 | 0.001 | None  |       |
    | 14  | mass of the back thigh                    | 1.5435146 | 0.001 | None  |       |
    | 15  | mass of the back shin                     | 1.5874476 | 0.001 | None  |       |
    | 16  | mass of the back foot                     | 1.0953975 | 0.001 | None  |       |
    | 17  | mass of the front thigh                   | 1.4380753 | 0.001 | None  |       |
    | 18  | mass of the front shin                    | 1.2008368 | 0.001 | None  |       |
    | 19  | mass of the front foot                    | 0.8845188 | 0.001 | None  |       |
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

        pos_noise = random.uniform(
            key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        )
        vel_noise = self._reset_noise_scale * random.normal(key, shape=(self.model.nv,))

        qpos = self.init_qpos + pos_noise
        qvel = self.init_qvel + vel_noise

        return mjx.forward(self.model, self.data.replace(qpos=qpos, qvel=qvel))

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
                mass[1:8],
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
        mass = mass.at[1:8].set(parameter[13:20])

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
            self._update_gym_env(gym_env, parameter)

        return gym_env

    def _update_gym_env(self, gym_env: MujocoEnv, parameter: jax.Array):
        model = gym_env.unwrapped.model
        model.geom_friction[0, :1] = parameter[:1]
        model.dof_armature[3:9] = parameter[1:7]
        model.dof_damping[3:9] = parameter[7:13]
        model.body_mass[1:8] = parameter[13:20]

    def _state_to_data(self, data: mjx.Data, states: jax.Array) -> mjx.Data:
        if self._exclude_current_positions_from_observation:
            states = jnp.concatenate([jnp.zeros(1), states])

        qpos = states[:9]
        qvel = states[9:]

        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jax.Array) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        if self._exclude_current_positions_from_observation:
            qpos = env_data.qpos[1:]
        else:
            qpos = env_data.qpos
        return jnp.concatenate([qpos, env_data.qvel])
