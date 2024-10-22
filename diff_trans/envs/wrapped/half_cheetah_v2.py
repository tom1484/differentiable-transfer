from typing import Dict, Tuple, List, cast

import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import jax
from jax import numpy as jnp
from jax import random

import mujoco
from mujoco import mjx

from .base import BaseEnv
from ... import envs, sim


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class HalfCheetah_v2(BaseEnv):
    """
    ### Description

    This environment is based on the work by P. Wawrzyński in
    ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
    The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8
    joints connecting them (including two paws). The goal is to apply a torque
    on the joints to make the cheetah run forward (right) as fast as possible,
    with a positive reward allocated based on the distance moved forward and a
    negative reward allocated for moving backward. The torso and head of the
    cheetah are fixed, and the torque can only be applied on the other 6 joints
    over the front and back thighs (connecting to the torso), shins
    (connecting to the thighs) and feet (connecting to the shins).

    ### Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied between *links*.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
    | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
    | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
    | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
    | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
    | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |


    ### Observation Space

    Observations consist of positional values of different body parts of the
    cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    By default, observations do not include the x-coordinate of the cheetah's center of mass. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 18 dimensions where the first dimension
    represents the x-coordinate of the cheetah's center of mass.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    will be returned in `info` with key `"x_position"`.

    However, by default, the observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:


    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
    | 3   | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angle (rad)              |
    | 4   | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
    | 5   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
    | 6   | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angle (rad)              |
    | 7   | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
    | 8   | x-coordinate of the front tip        | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | y-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
    | 12  | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
    | 13  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
    | 14  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |

    ### Rewards
    The reward consists of two parts:
    - *forward_reward*: A reward of moving forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (fixed to 5), where the frametime is 0.01 - making the
    default *dt = 5 * 0.01 = 0.05*. This reward would be positive if the cheetah
    runs forward (right).
    - *ctrl_cost*: A cost for penalising the cheetah if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 0.1

    The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ### Starting State
    All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,) with a noise added to the
    initial state for stochasticity. As seen before, the first 8 values in the
    state are positional and the last 9 values are velocity. A uniform noise in
    the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the positional values while a standard
    normal noise with a mean of 0 and standard deviation of `reset_noise_scale` is added to the
    initial velocity values of all zeros.

    ### Episode End
    The episode truncates when the episode length is greater than 1000.

    ### Arguments

    No additional arguments are currently supported in v2 and lower.

    ```
    env = gym.make('HalfCheetah-v2')
    ```

    v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default              | Description                                                                                                                                                       |
    | -------------------------------------------- | --------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"half_cheetah.xml"` | Path to a MuJoCo model                                                                                                                                            |
    | `forward_reward_weight`                      | **float** | `1.0`                | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see section on reward)                                                                                                             |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |

    ### Version History

    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
    ):
        env = envs.HalfCheetahConfig_v2()

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
    
    def _control_cost(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._ctrl_cost_weight * jnp.sum(jnp.square(actions), axis=1)

    def _step_wait(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        data = self._states
        control = self._actions

        qpos_before = jnp.asarray(data.qpos)
        data = sim.step_vj(self.env, self.env.model, data, control)
        self._states = data

        qpos = data.qpos
        observation = self.env._get_obs_vj(data)

        x_velocity = (qpos[:, 0] - qpos_before[:, 0]) / self.env.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._control_cost(control)
        reward = forward_reward - ctrl_cost

        done = jnp.zeros(self.num_env, dtype=bool)

        return observation, reward, done, [{}] * self.num_env
  