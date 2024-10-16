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
        self.mj_model = mj_model
        self.mj_data = mj_data

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

    def get_names(self, adr_list: list[int]) -> list[str]:
        raw_names = self.model.names
        names = []
        for adr in adr_list:
            adr_end = adr + 1
            while adr_end < len(raw_names) and raw_names[adr_end] != 0:
                adr_end += 1
            names.append(raw_names[adr:adr_end].decode())
            if adr_end >= len(raw_names):
                break
        
        return names
    
    def get_body_names(self) -> list[str]:
        return self.get_names(self.model.name_bodyadr)

    def get_actuator_names(self) -> list[str]:
        return self.get_names(self.model.name_actuatoradr)
    
    def get_joint_names(self) -> list[str]:
        return self.get_names(self.model.name_jntadr)

    def get_body_com(self, body_name: str) -> jnp.array:
        idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.subtree_com[idx]

    """
    Methods to be overwritten in the subclass
    """

    def reset(self, key: jnp.array) -> mjx.Data:
        NotImplementedError()

    def get_parameter(self) -> jnp.ndarray:
        NotImplementedError()

    def set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        NotImplementedError()

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        NotImplementedError()
