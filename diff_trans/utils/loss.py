from typing import List, Tuple

import jax
from jax import numpy as jnp

from diff_trans.envs import BaseDiffEnv
from diff_trans import sim
from diff_trans.utils.rollout import Trajectory, Transition


def extract_array_from_trajectories(
    trajectories: List[Trajectory],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    states = []
    next_states = []
    actions = []

    for trajectory in trajectories:
        for state, next_state, action, _, done in trajectory:
            if not done:
                states.append(state)
                next_states.append(next_state)
                actions.append(action)

    states = jnp.array(states)
    next_states = jnp.array(next_states)
    actions = jnp.array(actions)

    return states, next_states, actions


def extract_array_from_transitions(
    transitions: List[Transition],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    states = []
    next_states = []
    actions = []

    for state, next_state, action, _, done in transitions:
        if not done:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)

    states = jnp.array(states)
    next_states = jnp.array(next_states)
    actions = jnp.array(actions)

    return states, next_states, actions


def single_transition_loss(
    env: BaseDiffEnv,
    parameter: jax.Array,
    states: jax.Array,
    next_states: jax.Array,
    actions: jax.Array,
):
    """
    Compute the loss for a single transition.
    This only works under full observability.
    """
    model = env._set_parameter(parameter)
    data = env._state_to_data_v(env.data, states)
    data = sim.step_v(env, model, data, actions)
    next_states_sim = env._get_obs_v(data)

    diff: jax.Array = next_states - next_states_sim
    loss = jnp.mean(jnp.sum(jnp.abs(diff), axis=-1))

    return loss
