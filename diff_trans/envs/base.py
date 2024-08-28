from os import path

import jax
from jax import numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx


class EnvConfig:

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_dim: int,
    ):
        if model_path.startswith(".") or model_path.startswith("/"):
            fullpath = model_path
        elif model_path.startswith("~"):
            fullpath = path.expanduser(model_path)
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        mj_model = mujoco.MjModel.from_xml_path(fullpath)
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.data = mjx.put_data(mj_model, mj_data)

        self.init_qpos = jnp.array(self.data.qpos)
        self.init_qvel = jnp.array(self.data.qvel)

        self.state_dim = observation_dim
        ctrlrange = mj_model.actuator_ctrlrange.copy()
        low, high = ctrlrange.T
        self.control_range = (low, high)
        self.control_dim = ctrlrange.shape[0]

        self.frame_skip = frame_skip
        self.dt = frame_skip * mj_model.opt.timestep

        self.reset_vj = jax.jit(jax.vmap(self.reset))
        self._state_to_data_vj = jax.jit(
            jax.vmap(self._state_to_data, in_axes=(None, 0))
        )
        self._control_to_data_vj = jax.jit(
            jax.vmap(self._control_to_data, in_axes=(None, 0))
        )
        self._get_obs_vj = jax.jit(jax.vmap(self._get_obs))

    """
    Methods to be overwritten in the subclass
    """

    def reset(self, key: jnp.array) -> mjx.Data:
        NotImplementedError()

    def get_parameter(self) -> jnp.ndarray:
        NotImplementedError()

    def set_parameter(self, env_model: mjx.Model, parameter: jnp.ndarray) -> mjx.Model:
        NotImplementedError()

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _get_obs(self, env_data: mjx.Data) -> np.ndarray:
        NotImplementedError()
