from diff_trans.envs.gym import register

from .inverted_pendulum_v1 import DRInvertedPendulum_v1

register("DRInvertedPendulum-v5", DRInvertedPendulum_v1)
