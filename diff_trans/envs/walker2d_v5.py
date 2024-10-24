from typing import Tuple, Union

from jinja2 import pass_eval_context
import numpy as np
from jax import numpy as jnp
from jax import random, lax

from mujoco import mjx

from .base import BaseDiffEnv


class DiffWalker2d_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                      | Default   | Min  | Max  | Joint |
    |-----|--------------------------------|-----------|------|------|-------|
    | 0   | slide friction of the floor    | 1.0       | 0.5  | 1.5  | slide |
    | 1   | armature inertia of the hip1   | 1.0       | 0.5  | 1.5  | hinge |
    | 2   | armature inertia of the ankle1 | 1.0       | 0.5  | 1.5  | hinge |
    | 3   | armature inertia of the hip2   | 1.0       | 0.5  | 1.5  | hinge |
    | 4   | armature inertia of the ankle2 | 1.0       | 0.5  | 1.5  | hinge |
    | 5   | armature inertia of the hip3   | 1.0       | 0.5  | 1.5  | hinge |
    | 6   | armature inertia of the ankle3 | 1.0       | 0.5  | 1.5  | hinge |
    | 7   | armature inertia of the hip4   | 1.0       | 0.5  | 1.5  | hinge |
    | 8   | armature inertia of the ankle4 | 1.0       | 0.5  | 1.5  | hinge |
    | 9   | damping of the hip1            | 1.0       | 0.5  | 1.5  | hinge |
    | 10  | damping of the ankle1          | 1.0       | 0.5  | 1.5  | hinge |
    | 11  | damping of the hip2            | 1.0       | 0.5  | 1.5  | hinge |
    | 12  | damping of the ankle2          | 1.0       | 0.5  | 1.5  | hinge |
    | 13  | damping of the hip3            | 1.0       | 0.5  | 1.5  | hinge |
    | 14  | damping of the ankle3          | 1.0       | 0.5  | 1.5  | hinge |
    | 15  | damping of the hip4            | 1.0       | 0.5  | 1.5  | hinge |
    | 16  | damping of the ankle4          | 1.0       | 0.5  | 1.5  | hinge |
    | 17  | mass of the torso              | 0.3272492 | 0.16 | 0.48 |       |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
    ):
        observation_dim = 18

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
        self.parameter_range = jnp.array(
            [
                [
                    0.5,  # friction
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # armature
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # damping
                    0.16,  # mass
                ],
                [
                    1.5,  # friction
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  # armature
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  # damping
                    0.48,  # mass
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

    def get_parameter(self) -> jnp.ndarray:
        # friction = self.model.geom_friction.copy()
        # armature = self.model.dof_armature.copy()
        # damping = self.model.dof_damping.copy()
        # mass = self.model.body_mass.copy()

        # return jnp.concatenate(
        #     [
        #         friction[0, 0:1],
        #         armature[6:14],
        #         damping[6:14],
        #         mass[1:2],
        #     ]
        # )
        pass

    def set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        # friction = self.model.geom_friction
        # friction = friction.at[0, :1].set(parameter[:1])

        # armature = self.model.dof_armature
        # armature = armature.at[6:14].set(parameter[1:9])

        # damping = self.model.dof_damping
        # damping = damping.at[6:14].set(parameter[9:17])

        # mass = self.model.body_mass
        # mass = mass.at[1:2].set(parameter[17:18])

        # return self.model.replace(
        #     geom_friction=friction,
        #     dof_armature=armature,
        #     dof_damping=damping,
        #     body_mass=mass,
        # )
        pass

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        # qpos = states[:15]
        # qvel = states[15:29]

        # return data.replace(qpos=qpos, qvel=qvel)
        pass

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        qpos = data.qpos
        qvel = jnp.clip(data.qvel, -10, 10)

        # if self._exclude_current_positions_from_observation:
        #     qpos = qpos[1:]

        return jnp.concatenate([qpos, qvel])
