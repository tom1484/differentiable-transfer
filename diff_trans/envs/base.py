from typing import Optional

from os import path

import jax
from jax import numpy as jnp
import jax.core
import jax.lib
import numpy as np

import mujoco
from mujoco import mjx
from gymnasium import Env


class BaseDiffEnv:

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_dim: int,
    ):
        """
        Initialize the environment.
        """
        if model_path.startswith(".") or model_path.startswith("/"):
            fullpath = model_path
        elif model_path.startswith("~"):
            fullpath = path.expanduser(model_path)
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        # Load the MuJoCo model and data
        mj_model = mujoco.MjModel.from_xml_path(fullpath)
        mj_data = mujoco.MjData(mj_model)
        self.mj_model = mj_model
        self.mj_data = mj_data

        # Put the MuJoCo model and data into the JAX-compatible format
        self.model = mjx.put_model(mj_model)
        self.data = mjx.put_data(mj_model, mj_data)
        self.dt = float(self.model.opt.timestep) * frame_skip

        # Initialize the initial state
        self.init_qpos = jnp.array(self.data.qpos)
        self.init_qvel = jnp.array(self.data.qvel)

        # Initialize the state and control dimensions
        self.state_dim = observation_dim
        ctrlrange = mj_model.actuator_ctrlrange.copy()
        low, high = ctrlrange.T.astype(np.float32)
        self.control_range = (low, high)
        self.control_dim = ctrlrange.shape[0]

        # Initialize the frame skip and time step
        self.frame_skip = frame_skip
        self.dt = frame_skip * mj_model.opt.timestep

        # Initialize the JIT functions
        self.reset_vj = jax.jit(jax.vmap(self.reset))
        self._get_obs_vj = jax.jit(jax.vmap(self._get_obs))

        # Uncompiled functions
        self._get_obs_vj_ = jax.jit(jax.vmap(self._get_obs))
        self._state_to_data_vj_ = jax.jit(
            jax.vmap(self._state_to_data, in_axes=(None, 0))
        )
        self._control_to_data_vj_ = jax.jit(
            jax.vmap(self._control_to_data, in_axes=(None, 0))
        )

        # Prevent circular import
        from ..sim import step_vj

        self.step_vj = step_vj

        # Set default values
        self._reset_noise_scale = 0
        self.num_parameter = 0
        self.parameter_range = jnp.array([[0], [0]])

    def compile(self, num_envs: int):
        """
        Compile the JIT functions for the environment for the given number of environments.
        """
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, num_envs)
        self.reset_vj = self.reset_vj.lower(rng).compile()

        data = self.reset_vj(rng)
        self._get_obs_vj = self._get_obs_vj.lower(data).compile()

        control = jnp.zeros((num_envs, self.control_dim))
        # state = jnp.zeros((num_envs, self.state_dim))
        # self._state_to_data_vj = self._state_to_data_vj.lower(data, state).compile()
        # self._control_to_data_vj = self._control_to_data_vj.lower(
        #     data, control
        # ).compile()

        # Prevent circular import
        from ..sim import step_vj

        self.step_vj = step_vj.lower(self, self.model, data, control).compile()

    def _get_body_com(self, data: mjx.Data, idx: int) -> jnp.ndarray:
        """
        Get the center of mass of a body.
        """
        return data.subtree_com[idx]

    def _get_body_com_batch(self, data: mjx.Data, idx: int) -> jnp.ndarray:
        """
        Get the center of mass of a body for a batch of environments.
        """
        return data.subtree_com[:, idx]

    def _get_state_vector_batch(self, data: mjx.Data) -> jnp.ndarray:
        """
        Get the state vector of a batch of environments.
        """
        return jnp.concatenate([data.qpos, data.qvel], axis=1)

    def _get_names(self, adr_list: list[int]) -> list[str]:
        """
        Get the names of the given list of MuJoCo model addresses.
        Can be bodies, actuators, joints, or dof joints.
        """
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
        """
        Get the names of all bodies, actuators, joints, and dof joints.
        """
        raw_names = self.model.names.split(b"\0")
        names = []
        for name in raw_names:
            if name:
                names.append(name.decode())
        return names

    def get_body_names(self) -> list[str]:
        """
        Get the names of all bodies.
        """
        return self._get_names(self.model.name_bodyadr)

    def get_actuator_names(self) -> list[str]:
        """
        Get the names of all actuators.
        """
        return self._get_names(self.model.name_actuatoradr)

    def get_joint_names(self) -> list[str]:
        """
        Get the names of all joints.
        """
        return self._get_names(self.model.name_jntadr)

    def get_dof_joint_names(self) -> list[str]:
        """
        Get the names of all dof joints.
        """
        joint_names = self.get_joint_names()
        return [joint_names[joint_id] for joint_id in self.model.dof_jntid]

    """
    Methods to be overwritten in the subclass
    """

    def reset(self, key: jnp.array) -> mjx.Data:
        NotImplementedError()

    def _get_parameter(self) -> jnp.ndarray:
        NotImplementedError()

    def _set_parameter(self, parameter: jnp.ndarray) -> mjx.Model:
        NotImplementedError()

    def _create_gym_env(self, parameter: Optional[np.ndarray] = None, **kwargs) -> Env:
        NotImplementedError()

    def _state_to_data(self, data: mjx.Data, states: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _control_to_data(self, data: mjx.Data, control: jnp.ndarray) -> mjx.Data:
        NotImplementedError()

    def _get_obs(self, data: mjx.Data) -> np.ndarray:
        NotImplementedError()
