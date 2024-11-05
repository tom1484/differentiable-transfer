import numpy as np

from jax import numpy as jnp
from jax import random

from mujoco import mjx

from ..base import BaseDiffEnv


class FetchSlideConfig_v1(BaseDiffEnv):
    # TODO: Fill in the parameter range
    """
    ## Parameter Space

    | Num | Parameter                                 | Default   | Min | Max | Joint |
    |-----|-------------------------------------------|-----------|-----|-----|-------|
    | 0   | slide friction of the floor               | 0.4       | 0.2 | 0.6 | slide |
    """

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 0.01,
    ):
        observation_dim = 18
        super().__init__(
            "fetch/slide.xml",
            frame_skip,
            observation_dim,
        )

        self.reset_noise_scale = reset_noise_scale

        # TODO: Fill in the parameter range
        self.parameter_range = jnp.array([[], []])

    def reset(self, key: jnp.array) -> mjx.Data:
        # noise_low = -self.reset_noise_scale
        # noise_high = self.reset_noise_scale

        # pos_noise = random.uniform(
        #     key, shape=(self.model.nq,), minval=noise_low, maxval=noise_high
        # )
        # vel_noise = self.reset_noise_scale * random.normal(key, shape=(self.model.nv,))

        # qpos = self.init_qpos + pos_noise
        # qvel = self.init_qvel + vel_noise

        # return self.data.replace(qpos=qpos, qvel=qvel)
        pass

    def _get_parameter(self) -> jnp.ndarray:
        # friction = self.model.geom_friction.copy()
        # frictionloss = self.model.dof_frictionloss.copy()
        # armature = self.model.dof_armature.copy()
        # damping = self.model.dof_damping.copy()
        # mass = self.model.body_mass.copy()

        # return jnp.concatenate(
        #     [
        #         friction[0, 0:1],
        #         frictionloss[3:],
        #         armature[3:],
        #         damping[3:],
        #         mass[1:],
        #     ]
        # )
        pass

    def _set_parameter(self, env_model: mjx.Model, parameter: jnp.ndarray) -> mjx.Model:
        # friction = env_model.geom_friction
        # friction = friction.at[0, :1].set(parameter[:1])

        # frictionloss = env_model.dof_frictionloss
        # frictionloss = frictionloss.at[3:].set(parameter[1:7])

        # armature = env_model.dof_armature
        # armature = armature.at[3:].set(parameter[7:13])

        # damping = env_model.dof_damping
        # damping = damping.at[3:].set(parameter[13:19])

        # mass = env_model.body_mass
        # mass = mass.at[1:].set(parameter[19:26])

        # return env_model.replace(
        #     geom_friction=friction,
        #     dof_frictionloss=frictionloss,
        #     dof_armature=armature,
        #     dof_damping=damping,
        #     body_mass=mass,
        # )
        pass

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        # qpos = states[:9]
        # qvel = states[9:]
        # return data.replace(qpos=qpos, qvel=qvel)
        pass

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        # return data.replace(ctrl=control)
        pass

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        # return jnp.concatenate([env_data.qpos, env_data.qvel])
        pass
