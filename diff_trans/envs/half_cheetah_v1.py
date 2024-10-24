import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx

from .base import BaseDiffEnv


class DiffHalfCheetah_v1(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                                 | Default   | Min | Max | Joint |
    |-----|-------------------------------------------|-----------|-----|-----|-------|
    | 0   | slide friction of the floor               | 0.4       | 0.2 | 0.6 | slide |
    | 1   | friction loss of the back thigh rotor     | 0.0       | 0.0 | 0.3 | hinge |
    | 2   | friction loss of the back shin rotor      | 0.0       | 0.0 | 0.3 | hinge |
    | 3   | friction loss of the back foot rotor      | 0.0       | 0.0 | 0.3 | hinge |
    | 4   | friction loss of the front thigh rotor    | 0.0       | 0.0 | 0.3 | hinge |
    | 5   | friction loss of the front shin rotor     | 0.0       | 0.0 | 0.3 | hinge |
    | 6   | friction loss of the front foot rotor     | 0.0       | 0.0 | 0.3 | hinge |
    | 7   | armature inertia of the back thigh rotor  | 0.1       | 0.0 | 0.3 | hinge |
    | 8   | armature inertia of the back shin rotor   | 0.1       | 0.0 | 0.3 | hinge |
    | 9   | armature inertia of the back foot rotor   | 0.1       | 0.0 | 0.3 | hinge |
    | 10  | armature inertia of the front thigh rotor | 0.1       | 0.0 | 0.3 | hinge |
    | 11  | armature inertia of the front shin rotor  | 0.1       | 0.0 | 0.3 | hinge |
    | 12  | armature inertia of the front foot rotor  | 0.1       | 0.0 | 0.3 | hinge |
    | 13  | damping of the back thigh rotor           | 6.0       | 3.0 | 9.0 | hinge |
    | 14  | damping of the back shin rotor            | 4.5       | 3.0 | 6.0 | hinge |
    | 15  | damping of the back foot rotor            | 3.0       | 1.5 | 4.5 | hinge |
    | 16  | damping of the front thigh rotor          | 4.5       | 3.0 | 6.0 | hinge |
    | 17  | damping of the front shin rotor           | 3.0       | 1.5 | 4.5 | hinge |
    | 18  | damping of the front foot rotor           | 1.5       | 1.0 | 2.0 | hinge |
    | 19  | mass of the torso                         | 6.2502093 | 4.0 | 8.0 |       |
    | 20  | mass of the back thigh                    | 1.5435146 | 1.0 | 2.0 |       |
    | 21  | mass of the back shin                     | 1.5874476 | 1.0 | 2.0 |       |
    | 22  | mass of the back foot                     | 1.0953975 | 0.7 | 1.5 |       |
    | 23  | mass of the front thigh                   | 1.4380753 | 1.0 | 2.0 |       |
    | 24  | mass of the front shin                    | 1.2008368 | 0.8 | 1.6 |       |
    | 25  | mass of the front foot                    | 0.8845188 | 0.6 | 1.2 |       |
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

        # fmt: off
        self.parameter_range = jnp.array(
            [
                [
                    0.2,  # friction
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # friction loss
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # armature
                    3.0, 3.0, 1.5, 3.0, 1.5, 1.0,  # damping
                    4.0, 1.0, 1.0, 0.7, 1.0, 0.8, 0.6,  # mass
                ],
                [
                    0.6,  # friction 
                    0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # friction loss 
                    0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # armature 
                    9.0, 6.0, 4.5, 6.0, 4.5, 2.0,  # damping 
                    8.0, 2.0, 2.0, 1.5, 2.0, 1.6, 1.2,  # mass
                ],
            ]
        )
        # fmt: on

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        pos_noise = random.uniform(
            key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        )
        vel_noise = self.reset_noise_scale * random.normal(key, shape=(self.model.nv,))

        qpos = self.init_qpos + pos_noise
        qvel = self.init_qvel + vel_noise

        return mjx.step(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def get_parameter(self) -> jnp.ndarray:
        friction = self.model.geom_friction.copy()
        frictionloss = self.model.dof_frictionloss.copy()
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                friction[0, 0:1],
                frictionloss[3:],
                armature[3:],
                damping[3:],
                mass[1:],
            ]
        )

    def set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        friction = self.model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        frictionloss = self.model.dof_frictionloss
        frictionloss = frictionloss.at[3:].set(parameter[1:7])

        armature = self.model.dof_armature
        armature = armature.at[3:].set(parameter[7:13])

        damping = self.model.dof_damping
        damping = damping.at[3:].set(parameter[13:19])

        mass = self.model.body_mass
        mass = mass.at[1:].set(parameter[19:26])

        return self.model.replace(
            geom_friction=friction,
            dof_frictionloss=frictionloss,
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
