from os import path

import jax
from jax import numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx


class BaseDiffEnv:

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
        self.dt = float(self.model.opt.timestep) * frame_skip

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

        # default values
        self.reset_noise_scale = 0
        self.parameter_range = jnp.array([[0], [0]])

    def _get_body_com(self, data: mjx.Data, idx: int) -> jnp.ndarray:
        return data.subtree_com[idx]

    def _get_body_com_batch(self, data: mjx.Data, idx: int) -> jnp.ndarray:
        return data.subtree_com[:, idx]

    def _get_state_vector_batch(self, data: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate([data.qpos, data.qvel], axis=1)

    def _get_names(self, adr_list: list[int]) -> list[str]:
        raw_names = self.model.names
        names = []
        for adr in adr_list:
            adr_end = adr + 1
            while adr_end < len(raw_names) and raw_names[adr_end] != 0:
                adr_end += 1
            
            # Trim the trailing and leading null bytes
            name = raw_names[adr:adr_end].decode()
            name = name.strip("\x00")

            names.append(name)
            if adr_end >= len(raw_names):
                break
        
        return names
    
    def get_all_names(self) -> list[str]:
        raw_names = self.model.names.split(b"\0")
        names = []
        for name in raw_names:
            if name:
                names.append(name.decode())
        return names
    
    def get_body_names(self) -> list[str]:
        return self._get_names(self.model.name_bodyadr)

    def get_actuator_names(self) -> list[str]:
        return self._get_names(self.model.name_actuatoradr)
    
    def get_joint_names(self) -> list[str]:
        return self._get_names(self.model.name_jntadr)
    
    def get_dof_joint_names(self) -> list[str]:
        joint_names = self.get_joint_names()
        return [joint_names[joint_id] for joint_id in self.model.dof_jntid]
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
