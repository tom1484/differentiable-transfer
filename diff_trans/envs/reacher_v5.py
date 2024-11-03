import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx

from .base import BaseDiffEnv
from .utils.array import sidx


class DiffReacher_v5(BaseDiffEnv):
    """
    ## Parameter Space

    | Num | Parameter                  | Default   | Min   | Max   | Joint |
    |-----|----------------------------|-----------|-------|-------|-------|
    | 0   | armature inertia of joint0 | 1.0       | 0.5   | 1.5   | hinge |
    | 1   | armature inertia of joint1 | 1.0       | 0.5   | 1.5   | hinge |
    | 2   | damping of joint0          | 1.0       | 0.5   | 1.5   | hinge |
    | 3   | damping of joint1          | 1.0       | 0.5   | 1.5   | hinge |
    | 4   | mass of the arm0           | 0.0356047 | 0.018 | 0.056 |       |
    | 5   | mass of the arm1           | 0.0356047 | 0.018 | 0.056 |       |
    | 6   | mass of the fingertip      | 0.0041888 | 0.002 | 0.006 |       |
    """

    def __init__(self, frame_skip: int = 2):
        observation_dim = 10
        super().__init__(
            "reacher.xml",
            frame_skip,
            observation_dim,
        )

        # fmt: off
        self.parameter_range = jnp.array(
            [
                [
                    0.5, 0.5,  # armature
                    0.5, 0.5,  # damping
                    0.018, 0.018, 0.002,  # mass
                ],
                [
                    1.5, 1.5,  # armature
                    1.5, 1.5,  # damping
                    0.056, 0.056, 0.006,  # mass
                ],
            ]
        )
        # fmt: on

    def reset(self, key: jnp.array) -> mjx.Data:
        qpos = (
            random.uniform(key, shape=(self.model.nq,), minval=-0.1, maxval=0.1)
            + self.init_qpos
        )

        goal_angle = random.uniform(key, shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        goal_length = random.uniform(key, shape=(1,), minval=0.05, maxval=0.2)
        self.goal = (
            jnp.concatenate(
                [
                    jnp.cos(goal_angle),
                    jnp.sin(goal_angle),
                ]
            )
            * goal_length
        )

        qpos = qpos.at[-2:].set(self.goal)
        qvel = self.init_qvel + random.uniform(
            key, shape=(self.model.nv,), minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-2:].set(0)

        return mjx.step(self.model, self.data.replace(qpos=qpos, qvel=qvel))

    def get_parameter(self) -> jnp.ndarray:
        armature = self.model.dof_armature.copy()
        damping = self.model.dof_damping.copy()
        mass = self.model.body_mass.copy()

        return jnp.concatenate(
            [
                armature[:2],
                damping[:2],
                mass[1:4],
            ]
        )

    def set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        armature = self.model.dof_armature
        armature = armature.at[:2].set(parameter[:2])

        damping = self.model.dof_damping
        damping = damping.at[2:4].set(parameter[2:4])

        mass = self.model.body_mass
        mass = mass.at[1:4].set(parameter[4:7])

        return self.model.replace(
            dof_armature=armature,
            dof_damping=damping,
            body_mass=mass,
        )

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        # TODO: Use parallelized version
        cos = states[:2]
        sin = states[2:4]
        theta = jnp.arctan2(sin, cos)

        qpos_target = states[4:6]
        qvel_arm = states[6:8]

        qpos = data.qpos.at[:2].set(theta)
        qpos = qpos.at[2:].set(qpos_target)
        qvel = data.qvel.at[:2].set(qvel_arm)

        return data.replace(qpos=qpos, qvel=qvel)

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        return data.replace(ctrl=control)

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        theta = data.qpos[:2]
        return jnp.concatenate(
            [
                jnp.cos(theta),
                jnp.sin(theta),
                data.qpos[2:],
                data.qvel[:2],
                (self._get_body_com(data, 3) - self._get_body_com(data, 4))[:2],
            ]
        )
