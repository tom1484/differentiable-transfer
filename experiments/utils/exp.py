from typing import List, Optional, Tuple, Type, TypeVar, cast

from omegaconf import OmegaConf
from jax import numpy as jnp

from constants import ROOT_DIR
from .path import get_exp_file_levels, create_exp_assets

T = TypeVar("T")


def convert_arg_array(
    values: List[T], _type: Type[T], broadcast: Optional[int] = None
) -> jnp.ndarray:
    if type(values) is list:
        return jnp.array([_type(v) for v in values])
    elif broadcast is None:
        return jnp.array([_type(values)])
    else:
        return jnp.repeat(_type(values), broadcast)


C = TypeVar("C")


def load_config(
    exp_filepath: str,
    name: str,
    CONFIG: Type[C],
    config_path: Optional[str] = None,
) -> Tuple[Optional[C], List[str], str]:
    # Initialize default configuration
    default_config = OmegaConf.structured(CONFIG)

    # Create experiment assets (folders and default configuration)
    exp_levels = get_exp_file_levels("experiments", exp_filepath)
    new_config, default_config_path, models_dir = create_exp_assets(
        ROOT_DIR, exp_levels, name, default_config
    )

    if new_config:
        return None, exp_levels, models_dir

    if config_path is None:
        config_path = default_config_path

    # Load user-modified configuration
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(default_config, config)
    config = cast(CONFIG, config)

    return config, exp_levels, models_dir
