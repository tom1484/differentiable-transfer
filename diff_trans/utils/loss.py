from jax import numpy as jnp
from typing import List, Tuple

from diff_trans.envs import BaseDiffEnv
from diff_trans import sim


Trajectory = List[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
]


def single_transition_loss(env: BaseDiffEnv, parameter: jnp.ndarray, trajectories: List[Trajectory]):
    """
    Compute the loss for a single transition.
    This only works under full observability.
    """
    observations = []
    next_observations = []
    actions = []

    for trajectory in trajectories:
        for observation, next_observation, action, _, done in trajectory:
            if not done:
                observations.append(observation)
                next_observations.append(next_observation)
                actions.append(action)
    
    observations = jnp.array(observations)
    next_observations = jnp.array(next_observations)
    actions = jnp.array(actions)

    model = env.set_parameter(parameter)
    data = env.data
    # TODO: Do not use step_at_vj
    _, next_observations_sim = sim.step_at_vj(env, model, data, observations, actions)

    diff = next_observations - next_observations_sim
    loss = jnp.mean(jnp.sum(diff ** 2, axis=1))
    
    return loss