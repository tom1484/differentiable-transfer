from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np

from mujoco import mjx
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import make

from .base import BaseDiffEnv


class DiffHumanoid_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                                | Default   | Min   | Max  | Joint |
    |-----|------------------------------------------|-----------|-------|------|-------|
    | 0   | slide friction of the floor              | 1.0       | 0.001 | None | slide |
    | 1   | z-axis armature inertia of the abdomen   | 0.02      | 0.001 | None | hinge |
    | 2   | x-axis armature inertia of the abdomen   | 0.02      | 0.001 | None | hinge |
    | 3   | y-axis armature inertia of the abdomen   | 0.02      | 0.001 | None | hinge |
    | 4   | x-axis armature inertia of the right hip | 0.01      | 0.001 | None | hinge |
    | 5   | y-axis armature inertia of the right hip | 0.01      | 0.001 | None | hinge |
    | 6   | z-axis armature inertia of the right hip | 0.008     | 0.001 | None | hinge |
    | 7   | armature inertia of the right knee       | 0.006     | 0.001 | None | hinge |
    | 8   | x-axis armature inertia of the left hip  | 0.01      | 0.001 | None | hinge |
    | 9   | y-axis armature inertia of the left hip  | 0.01      | 0.001 | None | hinge |
    | 10  | z-axis armature inertia of the left hip  | 0.01      | 0.001 | None | hinge |
    | 11  | armature inertia of the left knee        | 0.006     | 0.001 | None | hinge |
    | 12  | armature inertia of the right shoulder1  | 0.0068    | 0.001 | None | hinge |
    | 13  | armature inertia of the right shoulder2  | 0.0051    | 0.001 | None | hinge |
    | 14  | armature inertia of the right elbow      | 0.0028    | 0.001 | None | hinge |
    | 15  | armature inertia of the left shoulder1   | 0.0068    | 0.001 | None | hinge |
    | 16  | armature inertia of the left shoulder2   | 0.0051    | 0.001 | None | hinge |
    | 17  | armature inertia of the left elbow       | 0.0028    | 0.001 | None | hinge |
    | 18  | z-axis damping of the abdomen            | 5.0       | 0.001 | None | hinge |
    | 19  | x-axis damping of the abdomen            | 5.0       | 0.001 | None | hinge |
    | 20  | y-axis damping of the abdomen            | 5.0       | 0.001 | None | hinge |
    | 21  | x-axis damping of the right hip          | 5.0       | 0.001 | None | hinge |
    | 22  | y-axis damping of the right hip          | 5.0       | 0.001 | None | hinge |
    | 23  | z-axis damping of the right hip          | 1.0       | 0.001 | None | hinge |
    | 24  | damping of the right knee                | 1.0       | 0.001 | None | hinge |
    | 25  | x-axis damping of the left hip           | 5.0       | 0.001 | None | hinge |
    | 26  | y-axis damping of the left hip           | 5.0       | 0.001 | None | hinge |
    | 27  | z-axis damping of the left hip           | 5.0       | 0.001 | None | hinge |
    | 28  | damping of the left knee                 | 1.0       | 0.001 | None | hinge |
    | 29  | x-axis damping of the right shoulder1    | 1.0       | 0.001 | None | hinge |
    | 30  | y-axis damping of the right shoulder2    | 1.0       | 0.001 | None | hinge |
    | 31  | z-axis damping of the right elbow        | 1.0       | 0.001 | None | hinge |
    | 32  | x-axis damping of the left shoulder1     | 1.0       | 0.001 | None | hinge |
    | 33  | y-axis damping of the left shoulder2     | 1.0       | 0.001 | None | hinge |
    | 34  | z-axis damping of the left elbow         | 1.0       | 0.001 | None | hinge |
    | 35  | mass of the torso                        | 8.907462  | 0.001 | None |       |
    | 36  | mass of the lwaist                       | 2.2619467 | 0.001 | None |       |
    | 37  | mass of the pelvis                       | 6.6161942 | 0.001 | None |       |
    | 38  | mass of the right thigh                  | 4.751751  | 0.001 | None |       |
    | 39  | mass of the right shin                   | 2.755696  | 0.001 | None |       |
    | 40  | mass of the right foot                   | 1.7671459 | 0.001 | None |       |
    | 41  | mass of the left thigh                   | 4.751751  | 0.001 | None |       |
    | 42  | mass of the left shin                    | 2.755696  | 0.001 | None |       |
    | 43  | mass of the left foot                    | 1.7671459 | 0.001 | None |       |
    | 44  | mass of the right upper arm              | 1.6610805 | 0.001 | None |       |
    | 45  | mass of the right lower arm              | 1.2295402 | 0.001 | None |       |
    | 46  | mass of the left upper arm               | 1.6610805 | 0.001 | None |       |
    | 47  | mass of the left lower arm               | 1.2295402 | 0.001 | None |       |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
    ):
        observation_dim = 47

        super().__init__(
            "humanoid.xml",
            frame_skip,
            observation_dim,
        )

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        # fmt: off
        self.num_parameter = 18
        self.parameter_range = jnp.array(
            [
                [ 0.001 ] * 48,
                [ jnp.inf ] * 48,
            ]
        )
        # fmt: on

    # def reset(self, key: jnp.array) -> mjx.Data:
    #     noise_low = -self._reset_noise_scale
    #     noise_high = self._reset_noise_scale

    #     qpos = self.init_qpos + random.uniform(
    #         key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
    #     )
    #     qvel = self.init_qvel + random.uniform(
    #         key, shape=(self.model.nv,), minval=noise_low, maxval=noise_high
    #     )

    #     return mjx.step(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def _get_parameter(self) -> jax.Array:
        friction = self.model.geom_friction.copy()
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                friction[0, 0:1],
                armature[6:23],
                damping[6:23],
                mass[1:14],
            ]
        )

    def _set_parameter(self, parameter: jax.Array) -> mjx.Model:
        friction = self.model.geom_friction
        friction = friction.at[0, :1].set(parameter[:1])

        armature = self.model.dof_armature
        armature = armature.at[6:23].set(parameter[1:18])

        damping = self.model.dof_damping
        damping = damping.at[6:23].set(parameter[18:35])

        mass = self.model.body_mass
        mass = mass.at[1:14].set(parameter[35:48])

        return self.model.replace(
            geom_friction=friction,
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )
    
    def _create_gym_env(
        self, parameter: Optional[np.ndarray] = None, **kwargs
    ) -> MujocoEnv:
        gym_env = make("Humanoid-v5", **kwargs)

        if parameter is not None:
            model = gym_env.unwrapped.model

            model.geom_friction[0, :1] = parameter[:1]
            model.dof_armature[6:23] = parameter[1:18]
            model.dof_damping[6:23] = parameter[18:35]
            model.body_mass[1:14] = parameter[35:48]

        return gym_env

    # def _state_to_data(self, data: mjx.Data, states: jax.Array) -> mjx.Data:
    #     qpos = states[:15]
    #     qvel = states[15:29]

    #     return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jax.Array) -> mjx.Data:
        return data.replace(ctrl=control)

    # def contact_forces(self, data: mjx.Data) -> jax.Array:
    #     raw_contact_forces = data.cfrc_ext.flatten()
    #     min_value, max_value = self._contact_force_range
    #     contact_forces = jnp.clip(raw_contact_forces, min_value, max_value)

    #     return contact_forces

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        qpos = data.qpos
        qvel = data.qvel

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:]
        else:
            com_inertia = jnp.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:]
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:]
        else:
            actuator_forces = jnp.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:]
        else:
            external_contact_forces = jnp.array([])

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate(
            [
                qpos,
                qvel,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            ]
        )
