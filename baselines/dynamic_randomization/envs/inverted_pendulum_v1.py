from diff_trans.envs.gym_wrapper import InvertedPendulum_v5
from .base import DREnv


class DRInvertedPendulum_v1(InvertedPendulum_v5, DREnv):
    def __init__(self, *args, eval_args=None, **kwargs):
        super(InvertedPendulum_v5, self).__init__(*args, **kwargs)
        self.eval_args = eval_args

    def sample_goal(self):
        if not self.eval_args or self.eval_args["goal_eval"] == "random":
            return self.random_sample_goal()
        elif not self.eval_args or self.eval_args["goal_eval"] == "oor-box":
            return self.out_of_reach_goal()
        else:
            goal = self.fixed_goal(self.eval_args["goal_pose"])
            if self.eval_args["start_eval"] == "constrained":
                self.constrained_start(goal)
            return goal
    
    def fixed_goal(self, goal):
        pass

    def random_sample_goal(self):
        pass
    
    def out_of_reach_goal(self):
        pass
    
    def constrained_start(self, goal):
        pass
    