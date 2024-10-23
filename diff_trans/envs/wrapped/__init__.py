from .base import BaseEnv
from .utils.entry import register, get_env

from .inverted_pendulum_v5 import InvertedPendulum_v1

# from .inverted_pendulum_v2 import InvertedPendulum_v2

register("InvertedPendulum-v5", InvertedPendulum_v1)
# register("InvertedPendulum-v2", InvertedPendulum_v2)


from .half_cheetah_v1 import HalfCheetah_v1
from .half_cheetah_v5 import HalfCheetah_v2

register("HalfCheetah-v1", HalfCheetah_v1)
register("HalfCheetah-v5", HalfCheetah_v2)


from .reacher_v5 import Reacher_v1

register("Reacher-v1", Reacher_v1)
