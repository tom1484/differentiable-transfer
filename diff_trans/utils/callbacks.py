import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, Dict, Any

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CallbackList

# try:
#     from tqdm import TqdmExperimentalWarning

#     # Remove experimental warning
#     warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
#     from tqdm.rich import tqdm
# except ImportError:
#     # Rich not installed, we only throw an error
#     # if the progress bar is used
#     tqdm = None


# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import (
#     DummyVecEnv,
#     VecEnv,
#     sync_envs_normalization,
# )

if TYPE_CHECKING:
    from stable_baselines3.common import base_class

from diff_trans.envs.wrapped import BaseEnv
from diff_trans.utils.rollout import evaluate_policy


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: BaseEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        callback_on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        # log_path: Optional[str] = None,
        # best_model_save_path: Optional[str] = None,
        # deterministic: bool = True,
        # render: bool = False,
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
        # self.deterministic = deterministic
        # self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        # if not isinstance(eval_env, VecEnv):
        #     eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        # self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        # if log_path is not None:
        #     log_path = os.path.join(log_path, "evaluations")
        # self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        # if self.best_model_save_path is not None:
        #     os.makedirs(self.best_model_save_path, exist_ok=True)
        # if self.log_path is not None:
        #     os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    # def _log_success_callback(
    #     self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    # ) -> None:
    #     """
    #     Callback passed to the  ``evaluate_policy`` function
    #     in order to log the success rate (when applicable),
    #     for instance when using HER.

    #     :param locals_:
    #     :param globals_:
    #     """
    #     info = locals_["info"]

    #     if locals_["done"]:
    #         maybe_is_success = info.get("is_success")
    #         if maybe_is_success is not None:
    #             self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            # if self.model.get_vec_normalize_env() is not None:
            #     try:
            #         sync_envs_normalization(self.training_env, self.eval_env)
            #     except AttributeError as e:
            #         raise AssertionError(
            #             "Training and eval env are not wrapped the same way, "
            #             "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
            #             "and warning above."
            #         ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            # TODO: Change this
            episode_rewards, episode_lengths = evaluate_policy(
                self.eval_env,
                self.model,
                n_eval_episodes=self.n_eval_episodes,
                # render=self.render,
                # deterministic=self.deterministic,
                return_episode_rewards=True,
                # warn=self.warn,
                # callback=self._log_success_callback,
            )

            mean_return, std_return = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_return)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_return:.2f} +/- {std_return:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                
            metrics = {
                "eval_mean_return": float(mean_return),
                "eval_std_return": float(std_return),
                "eval_mean_ep_length": mean_ep_length,
                "eval_std_ep_length": std_ep_length,
                "timesteps": self.num_timesteps
            }
            print(metrics)
            if self.callback_on_log is not None:
                self.callback_on_log(metrics)

            # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)

            # if len(self._is_success_buffer) > 0:
            #     success_rate = np.mean(self._is_success_buffer)
            #     if self.verbose >= 1:
            #         print(f"Success rate: {100 * success_rate:.2f}%")
            #     self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            # self.logger.record(
            #     "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            # )
            # self.logger.dump(self.num_timesteps)

            if mean_return > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                # if self.best_model_save_path is not None:
                #     self.model.save(
                #         os.path.join(self.best_model_save_path, "best_model")
                #     )
                self.best_mean_reward = float(mean_return)
                # Trigger callback on new best model, if needed
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
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalCallback

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training
