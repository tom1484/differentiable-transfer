from typing import Dict, Tuple, Union

from gymnasium.spaces import Box

from jax import numpy as jnp
import numpy as np

from .base import BaseEnv
from ... import envs, sim

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetah_v5(BaseEnv):
    r"""
    ## Description
    This environment is based on the work of P. Wawrzyński in ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
    The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws).
    The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
    The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).


    ## Action Space
    ```{figure} action_space_figures/half_cheetah.png
    :name: half_cheetah
    ```

    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
    | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
    | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
    | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
    | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
    | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (8 elements by default):* Position values of the robot's body parts.
    - *qvel (9 elements):* The velocities of these individual body parts (their derivatives).

    By default, the observation does not include the robot's x-coordinate (`rootx`).
    This can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (18,), float64)`, where the first observation element is the x-coordinate of the robot.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:


    | Num | Observation                                 | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the front tip               | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the front tip                      | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the back thigh                     | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
    | 3   | angle of the back shin                      | -Inf | Inf | bshin                            | hinge | angle (rad)              |
    | 4   | angle of the back foot                      | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
    | 5   | angle of the front thigh                    | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
    | 6   | angle of the front shin                     | -Inf | Inf | fshin                            | hinge | angle (rad)              |
    | 7   | angle of the front foot                     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
    | 8   | velocity of the x-coordinate of front tip   | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | velocity of the z-coordinate of front tip   | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angular velocity of the front tip           | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the back thigh          | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the back shin           | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the back foot           | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the front thigh         | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the front shin          | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the front foot          | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
    | excluded | x-coordinate of the front tip          | -Inf | Inf | rootx                            | slide | position (m)             |


    ## Rewards
    The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.

    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Half Cheetah moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime` which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Half Cheetah for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{9}, reset\_noise\_scale \times I_{9}]}$.
    The initial velocity state is $\mathcal{N}(0_{9}, reset\_noise\_scale^2 \times I_{9})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The Half Cheetah never terminates.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    HalfCheetah provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default              | Description                                                                                                                                                                                         |
    | -------------------------------------------- | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"half_cheetah.xml"` | Path to a MuJoCo model                                                                                                                                                                              |
    | `forward_reward_weight`                      | **float** | `1`                  | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                            |
    | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see `Rewards` section)                                                                                                                                               |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Restored the `xml_file` argument (was removed in `v4`).
        - Renamed `info["reward_run"]` to `info["reward_forward"]` to be consistent with the other environments.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen).
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        num_envs: int = 1,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        max_episode_steps: int = 1000,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        precompile: bool = True,
        **kwargs,
    ):
        diff_env = envs.DiffHalfCheetah_v5(
            frame_skip=frame_skip, reset_noise_scale=reset_noise_scale
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(diff_env.state_dim,),
            dtype=np.float64,
        )
        action_space = Box(
            low=diff_env.control_range[0],
            high=diff_env.control_range[1],
            dtype=np.float64,
        )

        super().__init__(
            diff_env=diff_env,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            observation_space=observation_space,
            action_space=action_space,
            default_camera_config=default_camera_config,
            precompile=precompile,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.diff_env.dt)),
        }

    def _control_cost(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._ctrl_cost_weight * jnp.sum(jnp.square(actions), axis=1)

    def _step_wait(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        data = self._states
        control = self._actions

        qpos_before = jnp.asarray(data.qpos)
        data = sim.step_vj(self.diff_env, self.diff_env.model, data, control)
        self._states = data

        qpos = data.qpos
        observation = self.diff_env._get_obs_vj(data)

        x_velocity = (qpos[:, 0] - qpos_before[:, 0]) / self.diff_env.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._control_cost(control)
        reward = forward_reward - ctrl_cost

        done = jnp.zeros(self.num_env, dtype=bool)

        return observation, reward, done, [{}] * self.num_env
