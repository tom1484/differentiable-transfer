from diff_trans.envs.wrapped import InvertedPendulum_v1
from .base import DREnv


class DRInvertedPendulum_v1(InvertedPendulum_v1, DREnv):
    def __init__(self, *args, **kwargs):
        super(InvertedPendulum_v1, self).__init__(*args, **kwargs)

    def sample_goal(self):
        return self.goal
    
    def fixed_goal(self, goal):
        return goal
    