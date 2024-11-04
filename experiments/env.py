import os
import enum
from typing import Any, List, Optional


class FlagType(enum.Enum):
    BOOL = 0
    INT = 1
    SWITCH = 2
    LIST = 3


def flag_str(v: Any, type: FlagType) -> str:
    if type == FlagType.BOOL:
        return "true" if v else "false"
    elif type == FlagType.INT:
        return 1 if v else 0
    elif type == FlagType.SWITCH:
        return "on" if v else "off"
    elif type == FlagType.LIST:
        return ",".join([str(i) for i in v])


def set_env_vars(
    xla_preallocate: bool = False,
    jax_debug_nans: bool = True,
    jax_traceback_filtering: bool = False,
    cuda_visible_devices: Optional[List[int]] = None,
):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = flag_str(
        xla_preallocate, FlagType.BOOL
    )
    os.environ["JAX_DEBUG_NANS"] = flag_str(jax_debug_nans, FlagType.BOOL)
    os.environ["JAX_TRACEBACK_FILTERING"] = flag_str(
        jax_traceback_filtering, FlagType.SWITCH
    )
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = flag_str(
            cuda_visible_devices, FlagType.LIST
        )
