from typing import Tuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax

from mujoco import mjx

from ..envs import BaseDiffEnv


def forward(model: mjx.Model, data: mjx.Data, num_steps: int) -> mjx.Data:

    def forward_single(data: mjx.Data, _):
        data = mjx.step(model, data)
        return data, None

    data, _ = lax.scan(forward_single, data, None, length=num_steps)
    return data


def step_at(
    env: BaseDiffEnv,
    model: mjx.Model,
    data: mjx.Data,
    state: jnp.ndarray,
    control: jnp.ndarray,
) -> Tuple[mjx.Data, jnp.ndarray]:
    data = env._state_to_data(data, state)
    data = env._control_to_data(data, control)

    num_steps = env.frame_skip
    data = forward(model, data, num_steps)

    return data, env._get_obs(data)


def step(
    env: BaseDiffEnv, model: mjx.Model, data: mjx.Data, control: jnp.ndarray
) -> mjx.Data:
    data = env._control_to_data(data, control)

    num_steps = env.frame_skip
    data = forward(model, data, num_steps)

    return data

step_at_v = jax.vmap(step_at, in_axes=(None, None, None, 0, 0))
step_v = jax.vmap(step, in_axes=(None, None, 0, 0))

step_at_vj = jax.jit(jax.vmap(step_at, in_axes=(None, None, None, 0, 0)), static_argnums=0)
step_vj = jax.jit(jax.vmap(step, in_axes=(None, None, 0, 0)), static_argnums=0)
