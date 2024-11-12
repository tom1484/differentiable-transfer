import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Dict, Any

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EventCallback

from diff_trans.envs.gym_wrapper import BaseEnv
from diff_trans.utils.rollout import evaluate_policy


class EvalCallback(EventCallback):
    def __init__(
        self,
        eval_env: BaseEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        callback_on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        reward_threshold: float = -np.inf,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self
        self.callback_on_log = callback_on_log

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.warn = warn

        self.reward_threshold = reward_threshold

        self.eval_env = eval_env
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []

            # TODO: Change this
            episode_rewards, episode_lengths = evaluate_policy(
                self.eval_env,
                self.model,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
            )

            mean_return, std_return = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_return)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}\n"
                    f"  episode_reward={mean_return:.2f} +/- {std_return:.2f}\n"
                    f"  episode_length={mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                )

            metrics = {
                "eval_mean_return": float(mean_return),
                "eval_std_return": float(std_return),
                "eval_mean_ep_length": mean_ep_length,
                "eval_std_ep_length": std_ep_length,
                "timesteps": self.num_timesteps,
            }
            if self.callback_on_log is not None:
                self.callback_on_log(metrics)

            if mean_return > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                self.best_mean_reward = float(mean_return)

                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class StopTrainingOnRewardThreshold(BaseCallback):
    parent: EvalCallback

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training


class MultiCallback(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.parent = self.parent
            callback.init_callback(self.model)

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = continue_training and callback.on_step()
        return continue_training


class SaveBestModelCallback(BaseCallback):
    def __init__(self, model_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.model_path = model_path

    def _on_step(self) -> bool:
        self.model.save(self.model_path)
        if self.verbose >= 1:
            print(f"Saved model to {self.model_path}")
        return True


class SaveModelCallback(BaseCallback):
    def __init__(self, folder: str, base_name: str = "model", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.folder = folder
        self.base_name = base_name

    def _on_step(self) -> bool:
        self.model.save(
            os.path.join(self.folder, f"{self.base_name}_{self.num_timesteps}")
        )
        return True
