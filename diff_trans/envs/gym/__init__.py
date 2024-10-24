from .utils.entry import register, get_env
from .base import BaseEnv

from .inverted_pendulum_v5 import InvertedPendulum_v1
from .half_cheetah_v1 import HalfCheetah_v1
from .half_cheetah_v5 import HalfCheetah_v2
from .reacher_v5 import Reacher_v1
from .ant_v5 import Ant_v5

register("InvertedPendulum-v5", InvertedPendulum_v1)
register("HalfCheetah-v1", HalfCheetah_v1)
register("HalfCheetah-v5", HalfCheetah_v2)
register("Reacher-v1", Reacher_v1)
register("Ant-v5", Ant_v5)