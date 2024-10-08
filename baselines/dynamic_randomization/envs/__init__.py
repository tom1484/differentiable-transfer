from diff_trans.envs.wrapped import register

from .inverted_pendulum_v1 import DRInvertedPendulum_v1

register("DRInvertedPendulum-v1", DRInvertedPendulum_v1)
