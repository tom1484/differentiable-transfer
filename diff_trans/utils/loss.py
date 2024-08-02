from jax import numpy as jnp
from typing import List, Tuple

from diff_trans.envs import EnvConfig
from diff_trans import sim


Trajectory = List[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
]


def single_transition_loss(env: EnvConfig, parameter: jnp.ndarray, trajectories: List[Trajectory]):
    observations = []
    next_observations = []
    actions = []

    for trajectory in trajectories:
        for observation, next_observation, action, reward, done in trajectory:
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
    
    observations = jnp.array(observations)
    next_observations = jnp.array(next_observations)
    actions = jnp.array(actions)

    model = env.set_parameter(env.model, parameter)
    datas = env._state_to_data_vj(env.data, observations)
    next_data = sim.step_vj(env, model, datas, actions)

    next_observations_sim = env._get_obs_vj(next_data)
    diff = next_observations - next_observations_sim
    loss = jnp.mean(jnp.sum(diff ** 2, axis=-1))
    
    return loss