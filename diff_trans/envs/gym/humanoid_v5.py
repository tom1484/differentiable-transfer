from typing import Tuple, Union

import numpy as np

from gymnasium.spaces import Box

from jax import numpy as jnp
from mujoco import mjx

from .base import BaseEnv
from ... import envs, sim


class Humanoid_v5(BaseEnv):
    """
    ## Description
    This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel in ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).
    The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts.
    The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

    Note: Although the robot is called "Ant", it is actually 75cm tall and weighs 910.88g, with the torso being 327.25g and each leg being 145.91g.

    ## Action Space
    ```{figure} action_space_figures/ant.png
    :name: ant
    ```

    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (13 elements by default):* Position values of the robot's body parts.
    - *qvel (14 elements):* The velocities of these individual body parts (their derivatives).
    - *cfrc_ext (78 elements):* This is the center of mass based external forces on the body parts.
    It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
    (external forces - force x, y, z and torque x, y, z)

    By default, the observation does not include the x- and y-coordinates of the torso.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (107,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (105,), float64)`, where the position and velocity elements are as follows:

    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Type (Unit)              |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | root                                   | free  | position (m)             |
    | 1   | w-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 2   | x-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 3   | y-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 4   | z-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 26  | angular velocity of the angle between back right links       | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | excluded | x-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
    | excluded | y-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |

    The body parts are:

    | body part                 | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
    |  -----------------------  |  ---   |  ---  |
    | worldbody (note: all values are constant 0) | 0  |excluded|
    | torso                     | 1  |0       |
    | front_left_leg            | 2  |1       |
    | aux_1 (front left leg)    | 3  |2       |
    | ankle_1 (front left leg)  | 4  |3       |
    | front_right_leg           | 5  |4       |
    | aux_2 (front right leg)   | 6  |5       |
    | ankle_2 (front right leg) | 7  |6       |
    | back_leg (back left leg)  | 8  |7       |
    | aux_3 (back left leg)     | 9  |8       |
    | ankle_3 (back left leg)   | 10 |9       |
    | right_back_leg            | 11 |10      |
    | aux_4 (back right leg)    | 12 |11      |
    | ankle_4 (back right leg)  | 13 |12      |

    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).


    **Note:**
    When using Ant-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
    Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the Ant environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).


    ## Rewards
    The total reward is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.

    - *healthy_reward*:
    Every timestep that the Ant is healthy (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $1$).
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Ant moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Ant for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
    - *contact_cost*:
    A negative reward to penalize the Ant if the external contact forces are too large.
    $w_{contact} \times \|F_{contact}\|_2^2$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
    $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on Observation Space).

    `info` contains the individual reward terms.

    But if `use_contact_forces=False` on `v4`
    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.


    ## Starting State
    The initial position state is $[0.0, 0.0, 0.75, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{15}, reset\_noise\_scale \times I_{15}]}$.
    The initial velocity state is $\mathcal{N}(0_{14}, reset\_noise\_scale^2 \times I_{14})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the z- and x-coordinates are non-zero so that the ant can immediately stand up and face forward (x-axis).


    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Ant is unhealthy.
    the Ant is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[0.2, 1.0]$).

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Ant provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Ant-v5', ctrl_cost_weight=0.5, ...)
    ```

    | Parameter                                  | Type       | Default      |Description                    |
    |--------------------------------------------|------------|--------------|-------------------------------|
    |`xml_file`                                  | **str**    | `"ant.xml"`  | Path to a MuJoCo model                                                                                                                                                                                      |
    |`forward_reward_weight`                     | **float**  | `1`          | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`ctrl_cost_weight`                          | **float**  | `0.5`        | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    |`contact_cost_weight`                       | **float**  | `5e-4`       | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    |`healthy_reward`                            | **float**  | `1`          | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`main_body`                                 |**str\|int**| `1`("torso") | Name or ID of the body, whose displacement is used to calculate the *dx*/_forward_reward_ (useful for custom MuJoCo models) (see `Rewards` section)                                                         |
    |`terminate_when_unhealthy`                  | **bool**   | `True`       | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                             |
    |`healthy_z_range`                           | **tuple**  | `(0.2, 1)`   | The ant is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                                 |
    |`contact_force_range`                       | **tuple**  | `(-1, 1)`    | Contact forces are clipped to this range in the computation of *contact_cost* (see `Rewards` section)                                                                                                       |
    |`reset_noise_scale`                         | **float**  | `0.1`        | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    |`exclude_current_positions_from_observation`| **bool**   | `True`       | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
    |`include_cfrc_ext_in_observation`           | **bool**   | `True`       | Whether to include *cfrc_ext* elements in the observations (see `Observation State` section)                                                                                                                |
    |`use_contact_forces` (`v4` only)            | **bool**   | `False`      | If `True`, it extends the observation space by adding contact forces (see `Observation Space` section) and includes contact_cost to the reward function (see `Rewards` section)                             |
    """

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
    ):
        env = envs.DiffHumanoid_v5()

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.state_dim,),
            dtype=np.float32,
        )
        action_space = Box(
            low=env.control_range[0], high=env.control_range[1], dtype=np.float32
        )

        super().__init__(
            num_envs, env, max_episode_steps, observation_space, action_space
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
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

    def healthy_reward(self, data: mjx.Data) -> jnp.ndarray:
        return self.is_healthy(data) * self._healthy_reward

    def control_cost(self, control: jnp.ndarray) -> jnp.ndarray:
        control_cost = self._ctrl_cost_weight * jnp.sum(control**2, axis=1)
        return control_cost

    def contact_forces(self, data: mjx.Data) -> jnp.ndarray:
        raw_contact_forces = data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = jnp.clip(raw_contact_forces, min_value, max_value)

        return contact_forces

    def contact_cost(self, data: mjx.Data) -> jnp.ndarray:
        contact_cost = self._contact_cost_weight * jnp.sum(
            jnp.square(self.contact_forces(data))
        )
        return contact_cost

    def is_healthy(self, data: mjx.Data) -> jnp.ndarray:
        state = self.get_state_vector(data)
        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.logical_and(
            jnp.isfinite(state).all(axis=1),
            jnp.logical_and(
                min_z <= state[:, 2],
                state[:, 2] <= max_z,
            ),
        )

        return is_healthy

    def _get_reward(
        self, data: mjx.Data, x_velocity: jnp.ndarray, control: jnp.ndarray
    ) -> jnp.ndarray:
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward(data)
        reward = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(control)
        contact_cost = self.contact_cost(data)
        costs = ctrl_cost + contact_cost

        reward -= costs
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _step_wait(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        data = self._states
        control = self._actions

        main_pos_before = self.diff_env._get_body_com_batch(data, self._main_body)
        data = sim.step_vj(self.diff_env, self.diff_env.model, data, control)
        self._states = data

        main_pos_after = self.diff_env._get_body_com_batch(data, self._main_body)
        main_velocity = (main_pos_after - main_pos_before) / self.diff_env.dt
        x_velocity = main_velocity[:, 0]

        observation = self.diff_env._get_obs_vj(data)
        reward, reward_info = self._get_reward(data, x_velocity, control)

        info = {
            # TODO: add more info
            **reward_info,
        }

        terminated = jnp.logical_and(
            jnp.logical_not(self.is_healthy(data)), self._terminate_when_unhealthy
        )

        return observation, reward, terminated, info
