import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx

from .base import EnvConfig


class InvertedPendulumConfig_v1(EnvConfig):
    """
    ## Parameter Space

    | Num | Parameter                         | Default   | Min | Max  | Joint |
    |-----|-----------------------------------|-----------|-----|------|-------|
    | 0   | friction loss of the slider       | 0.0       | 0.0 | 0.5  | slide |
    | 1   | friction loss of the pole rotor   | 0.0       | 0.0 | 0.5  | hinge |
    | 2   | armature mass of the cart         | 0.0       | 0.0 | 2.0  | slide |
    | 3   | armature inertia of the pole      | 0.0       | 0.0 | 1.0  | hinge |
    | 4   | damping of the slider             | 1.0       | 0.5 | 1.5  | slide |
    | 5   | damping of the pole rotor         | 1.0       | 0.5 | 1.5  | hinge |
    | 6   | mass of the cart                  | 10.471975 | 5.0 | 15.0 |       |
    | 7   | mass of the joint                 | 5.0185914 | 2.5 | 7.5  |       |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 0.02,
    ):
        observation_dim = 4
        super().__init__("inverted_pendulum.xml", frame_skip, observation_dim)

        self.reset_noise_scale = reset_noise_scale

        self.parameter_range = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 5.0, 2.5],
                [0.5, 0.5, 2.0, 1.0, 1.5, 1.5, 15.0, 7.5],
            ]
        )

    def reset(self, key: jnp.array) -> mjx.Data:
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        noise = random.uniform(
            key, shape=(2 * self.model.nq,), minval=noise_low, maxval=noise_high
        )

        qpos = self.init_qpos + noise[:2]
        qvel = self.init_qvel + noise[2:]

        return mjx.step(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def get_parameter(self) -> jnp.ndarray:
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

    def set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
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

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        qpos = states[:2]
        qvel = states[2:]
        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        return jnp.concatenate([env_data.qpos, env_data.qvel])
