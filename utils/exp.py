from jax import numpy as jnp
from typing import List, Optional, Type, TypeVar

T = TypeVar("T")


def convert_arg_array(
    values: List[T], type: Type[T], broadcast: Optional[int] = None
) -> jnp.ndarray:
    if type(values) is list:
        return jnp.array([type(v) for v in values])
    elif broadcast is None:
        return jnp.array([type(values)])
    else:
        return jnp.repeat(type(values), broadcast)
