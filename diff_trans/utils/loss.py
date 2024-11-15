from jax import numpy as jnp
from typing import List

from diff_trans.envs import BaseDiffEnv
from diff_trans import sim
from diff_trans.utils.rollout import Transition


def single_transition_loss(
    env: BaseDiffEnv, parameter: jnp.ndarray, transitions: List[Transition]
):
    """
    Compute the loss for a single transition.
    This only works under full observability.
    """
    observations = []
    next_observations = []
    actions = []

    for observation, next_observation, action, _, done in transitions:
        if not done:
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)

    observations = jnp.array(observations)
    next_observations = jnp.array(next_observations)
    actions = jnp.array(actions)

    model = env._set_parameter(parameter)
    data = env._state_to_data_vj_(env.data, observations)
    data = sim.step_vj(env, model, data, actions)
    next_observations_sim = env._get_obs_vj_(data)

    diff = next_observations - next_observations_sim
    loss = jnp.mean(jnp.sum(diff**2, axis=1))

    return loss
