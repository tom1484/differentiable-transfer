from typing import List, Optional, Tuple, Type, TypeVar, cast, Any

import os
from omegaconf import OmegaConf
from jax import numpy as jnp

from constants import ROOT_DIR
from .path import get_exp_file_levels

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


def create_exp_assets(
    root: str, exp_levels: List[str], exp_name: str, default_config: Any
) -> Tuple[bool, str, str]:
    configs_dir = os.path.join(root, "configs", *exp_levels)
    config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
    os.makedirs(configs_dir, exist_ok=True)

    config_exists = os.path.exists(config_path)
    if not config_exists:
        with open(config_path, "w") as config_file:
            OmegaConf.save(default_config, config_file)

    models_dir = os.path.join(root, "models", *exp_levels, exp_name)
    os.makedirs(models_dir, exist_ok=True)

    return not config_exists, config_path, models_dir


C = TypeVar("C")


def load_config(
    exp_filepath: str,
    name: str,
    Config: Type[C],
    config_path: Optional[str] = None,
) -> Tuple[Optional[C], List[str], str]:
    # Initialize default configuration
    default_config = OmegaConf.structured(Config)

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
    config = cast(Config, config)

    return config, exp_levels, models_dir
