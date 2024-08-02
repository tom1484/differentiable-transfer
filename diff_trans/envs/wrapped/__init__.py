from .base import BaseEnv

from .inverted_pendulum import InvertedPendulum
from .half_cheetah import HalfCheetah

from .utils.entry import register, get_env


register("InvertedPendulum-v1", InvertedPendulum)
register("HalfCheetah-v1", HalfCheetah)
