import os
import enum
import jax


class FlagType(enum.Enum):
    BOOL = 0
    INT = 1
    SWITCH = 2


def flag_str(v: bool, type: FlagType) -> str:
    if type == FlagType.BOOL:
        return "true" if v else "false"
    elif type == FlagType.INT:
        return 1 if v else 0
    elif type == FlagType.SWITCH:
        return "on" if v else "off"


def set_jax_config(
    preallocate: bool = False,
    debug_nans: bool = True,
    traceback_filtering: bool = False,
):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = flag_str(preallocate, FlagType.BOOL)
    os.environ["JAX_DEBUG_NANS"] = flag_str(debug_nans, FlagType.BOOL)
    os.environ["JAX_TRACEBACK_FILTERING"] = flag_str(traceback_filtering, FlagType.SWITCH)
