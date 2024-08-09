import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx

from .base import EnvConfig
from .utils.array import sidx


class HalfCheetahConfig_v2(EnvConfig):
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
        frame_skip: int = 2,
        reset_noise_scale: float = 0.1,
    ):
        observation_dim = 18
        super().__init__(
            "half_cheetah.xml",
            frame_skip,
            observation_dim,
        )

        self.reset_noise_scale = reset_noise_scale

        self.parameter_range = jnp.array(
            [[0.2, 0.0, 0.0, 3.0, 3.0, 4.0], [0.6, 0.3, 0.3, 9.0, 6.0, 8.0]]
        )

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        pos_noise = random.uniform(
            key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        )
        vel_noise = self.reset_noise_scale * random.normal(key, shape=(self.model.nv,))

        qpos = self.init_qpos + pos_noise
        qvel = self.init_qvel + vel_noise

        return self.data.replace(qpos=qpos, qvel=qvel)

    def get_parameter(self) -> jnp.ndarray:
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

    def set_parameter(self, env_model: mjx.Model, parameter: jnp.ndarray) -> mjx.Model:
        friction = env_model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        armature = env_model.dof_armature
        armature = armature.at[sidx(3, 6)].set(parameter[1:3])

        damping = env_model.dof_damping
        damping = damping.at[sidx(3, 6)].set(parameter[3:5])

        mass = env_model.body_mass
        mass = mass.at[1:2].set(parameter[5:6])

        return env_model.replace(
            geom_friction=friction,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        qpos = states[:9]
        qvel = states[9:]
        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        return jnp.concatenate([env_data.qpos, env_data.qvel])
        # return jnp.concatenate([env_data.qpos[1:], env_data.qvel])
