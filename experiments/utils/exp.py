from typing import List, Optional, Tuple, Type, TypeVar, cast

import json
from jax import numpy as jnp

from constants import ROOT_DIR
from .path import get_exp_file_levels, create_exp_assets

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


C = TypeVar("C")


def load_config(
    exp_filepath: str,
    name: str,
    CONFIG: Type[C],
    config_path: Optional[str] = None,
) -> Tuple[Optional[C]]:
    # Initialize default configuration
    default_config = CONFIG()

    # Create experiment assets (folders and default configuration)
    exp_levels = get_exp_file_levels("experiments", exp_filepath)
    new_config, default_config_path, models_dir = create_exp_assets(
        ROOT_DIR, exp_levels, name, default_config.to_dict()
    )

    if new_config:
        return None, exp_levels, models_dir

    if config_path is None:
        config_path = default_config_path

    # Load user-modified configuration
    config_file = open(config_path, "r")
    config_dict = json.load(config_file)
    config_file.close()

    config = cast(CONFIG, CONFIG.from_dict(config_dict))
    return config, exp_levels, models_dir
