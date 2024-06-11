import math
import os

from typing import Optional

import numpy as np

import warp as wp
import warp.sim as wsim
import warp.sim.render as wrender

import gymnasium as gym

from .warp_env import WarpEnv
from ..utils.path import get_assets_path


@wp.kernel
def extract_parameter(
    joint_armature: wp.array(dtype=wp.float32),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    body_mass: wp.array(dtype=wp.float32),
    parameter: wp.array(dtype=wp.float32),
):
    parameter[0] = joint_armature[0]
    parameter[1] = joint_armature[1]
    parameter[2] = joint_linear_compliance[0]
    parameter[3] = joint_angular_compliance[0]
    parameter[4] = body_mass[1]
    parameter[5] = body_mass[2]


@wp.kernel
def insert_parameter(
    parameter: wp.array(dtype=wp.float32),
    joint_armature: wp.array(dtype=wp.float32),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    body_mass: wp.array(dtype=wp.float32),
):
    joint_armature[0] = parameter[0]
    joint_armature[1] = parameter[1]
    joint_linear_compliance[0] = parameter[2]
    joint_angular_compliance[0] = parameter[3]
    body_mass[1] = parameter[4]
    body_mass[2] = parameter[5]


class InvertedPendulumWarpEnv(WarpEnv):
    def __init__(
        self,
        num_envs=1,
        fps=60,
        sim_substeps=10,
        requires_grad=False,
        store_history=True,
        urdf_path=None,
    ):
        super().__init__(
            fps=fps,
            sim_substeps=sim_substeps,
        )

        self.store_history = store_history
        self.requires_grad = requires_grad

        self.num_envs = num_envs

        articulation_builder = wsim.ModelBuilder(up_vector=wp.vec3(0.0, 0.0, 1.0))
        urdf_path = urdf_path or os.path.join(
            get_assets_path(), "inverted_pendulum.xml"
        )
        wsim.parse_urdf(
            urdf_path,
            articulation_builder,
            xform=wp.transform_identity(),
            density=100.0,
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        builder = wsim.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform_identity(),
                # xform=wp.transform(np.array((i * 2.0, 4.0, 0.0)), wp.quat_identity()),
            )

        # finalize model
        self.model = builder.finalize(requires_grad=self.requires_grad)
        self.model.ground = False

        self.integrator = wsim.FeatherstoneIntegrator(self.model)

        self.state_size = self.model.joint_q.shape[0] + self.model.joint_qd.shape[0]
        self.control_size = self.model.joint_act.shape[0]

    @property
    def state(self) -> wsim.model.State:
        return self.states[self.sim_tick]

    @state.setter
    def state(self, value: wsim.model.State):
        self.states[self.sim_tick] = value

    @property
    def parameter(self) -> wp.array:
        parameter = wp.array(np.zeros((6,)), dtype=wp.float32, requires_grad=True)
        wp.launch(
            extract_parameter,
            dim=1,
            inputs=(
                self.model.joint_armature,
                self.model.joint_linear_compliance,
                self.model.joint_angular_compliance,
                self.model.body_mass,
            ),
            outputs=(parameter,),
        )

        return parameter

    @parameter.setter
    def parameter(self, value: np.ndarray | wp.array):
        if type(value) is np.ndarray:
            value = wp.array(value, dtype=wp.float32, requires_grad=True)
        
        wp.launch(
            insert_parameter,
            dim=1,
            inputs=(value,),
            outputs=(
                self.model.joint_armature,
                self.model.joint_linear_compliance,
                self.model.joint_angular_compliance,
                self.model.body_mass,
            ),
        )

    def reset(self, init_state: Optional[np.ndarray] = None):
        if init_state is not None:
            init_state = init_state.reshape((self.num_envs, self.state_size))
        else:
            init_state = np.zeros((self.num_envs, self.state_size), dtype=np.float32)

        state = self.model.state()

        init_joint_q = init_state[:, :3].flatten()
        init_joint_qd = init_state[:, 3:].flatten()

        state.joint_q = wp.array(init_joint_q, dtype=wp.float32, requires_grad=True)
        state.joint_qd = wp.array(init_joint_qd, dtype=wp.float32, requires_grad=True)

        # TODO: Figure out why this doesn't work
        # self.model.joint_q = wp.array(init_joint_q, dtype=wp.float32, requires_grad=True)
        # self.model.joint_qd = wp.array(init_joint_qd, dtype=wp.float32, requires_grad=True)
        # state = self.model.state()

        self.control = self.model.control()
        self.states = [state]

        self.sim_tick = 0
        self.sim_time = 0.0

    def forward(self, control: np.ndarray):
        self.control.joint_act = wp.array(control, dtype=wp.float32)

        state = self.state
        for _ in range(self.sim_substeps):
            state.clear_forces()
            state_next = self.model.state()

            self.integrator.simulate(
                self.model, state, state_next, self.sim_dt, control=self.control
            )

            state = state_next

        if self.store_history:
            self.states.append(state)
            self.sim_tick += 1
        else:
            self.state = state

        self.control.reset()

    def step(self, control: Optional[np.ndarray] = None):
        if control is None:
            control = np.zeros(self.num_envs * self.control_size, dtype=np.float32)
        else:
            assert control.shape == (self.num_envs * self.control_size,)

        self.forward(control)
        self.sim_time += self.frame_dt

    def close(self):
        pass

    def render(self, path: str, scaling=1.0):
        renderer = wrender.SimRenderer(
            path=path, model=self.model, scaling=scaling, fps=self.fps
        )

        with wp.ScopedTimer("render"):
            for i in range(len(self.states)):
                self.state = self.states[i]
                self.sim_time = i * self.sim_dt
                renderer.begin_frame(self.sim_time)
                renderer.render(self.state)
                renderer.end_frame()

        renderer.save()


class InvertedPendulumEnv(gym.Env):
    """
    Tunable parameters:
    - cart joint armature
    - pole joint armature
    - cart joint damping
    - pole joint damping
    - cart mass
    - pole mass
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        fps=60,
        substeps=10,
        urdf_path=None,
    ):
        self.warp_env = InvertedPendulumWarpEnv(
            num_envs=1,
            fps=fps,
            sim_substeps=substeps,
            urdf_path=urdf_path,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(4,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ready = False
    
    def get_parameter(self):
        return self.warp_env.parameter
    
    def set_parameter(self, parameter: np.ndarray):
        self.warp_env.parameter = parameter

    def _wrap_angle(self, angle: np.float32):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def _sim_state_to_observation(self, sim_state):
        joint_q = sim_state.joint_q.numpy()
        joint_qd = sim_state.joint_qd.numpy()

        joint_q.squeeze()
        joint_qd.squeeze()

        return np.hstack([joint_q, joint_qd])

    def _action_to_sim_control(self, action: np.ndarray):
        control = np.zeros(self.warp_env.control_size, dtype=np.float32)
        # control[0:1] = np.array(action).squeeze()
        control[1:] = np.array(action).squeeze()

        return control

    def _test_initial_state(self, init_state: np.ndarray):
        self.warp_env.reset(init_state)
        self.warp_env.step(np.zeros((2,)))
        state = self._sim_state_to_observation(self.warp_env.state)

        # Make sure the kinematics are stable
        if np.isnan(state).any() or np.any(np.abs(state - init_state) > 0.2):
            return False

        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Randomize initial state
        init_state = np.zeros((4,))
        while True:
            init_state = np.random.uniform(-0.05, 0.05, size=(4,))
            if self._test_initial_state(init_state):
                self.warp_env.reset(init_state)
                break

        # Get initial observation
        sim_state = self.warp_env.state
        observation = self._sim_state_to_observation(sim_state)

        self.ready = True

        return observation, {}

    def step(self, action: np.ndarray):
        if not self.ready:
            raise ValueError("Must call reset() before step()")

        # Apply action to the simulation
        control = self._action_to_sim_control(action)
        self.warp_env.step(control)
        sim_state = self.warp_env.state

        # Get observation, reward, termination, and info
        observation = self._sim_state_to_observation(sim_state)
        reward = 1.0
        terminated = np.abs(observation[1]) > 0.2
        truncated = False

        if terminated:
            self.ready = False

        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def render(self):
        pass
