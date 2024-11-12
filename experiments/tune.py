from typing import List, Optional, Dict, Any, Type, Tuple, Callable

import typer
from diff_trans.envs.base import BaseDiffEnv
from experiments.utils.config import *

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class TrainConfig:
    num_envs: int = 1
    max_timesteps: int = 100000
    use_checkpoint: bool = False


@dataclass
class TuneConfig:
    stop_tuning_below: float = 1.0
    tune_frequency: int = 1000
    learning_rate: float = 1e-2
    gradient_steps: int = 1

    start_rollout_at: int = 10000
    max_rollouts: int = 10
    rollout_frequency: int = 5000
    rollout_timesteps: int = 500


@dataclass
class EvalConfig:
    num_episodes: int = 256
    frequency: int = 10000


@dataclass
class Config:
    cuda_visible_devices: Optional[List[str]] = None

    num_exp: int = 1
    verbose: int = 1

    algorithm: str = "SAC"
    algorithm_config: Optional[Dict[str, Any]] = None

    env_name: str = "InvertedPendulum-v5"
    env_config: Optional[Dict[str, Any]] = None

    param_ids: Optional[List[int]] = None
    param_values: Optional[List[float]] = None
    freeze_ids: Optional[List[int]] = None
    freeze_all: bool = False

    train: TrainConfig = TrainConfig()
    tune: TuneConfig = TuneConfig()
    eval: EvalConfig = EvalConfig()

    log_wandb: bool = True
    debug_nans: bool = False


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    config_path: Optional[str] = typer.Option(
        None, help="Path to the configuration JSON file"
    ),
):
    import os
    from experiments.utils.exp import load_config

    config, exp_levels, models_dir = load_config(
        __file__, name, Config, config_path=config_path
    )
    if config is None:
        print("Configuration created")
        return

    from experiments.env import set_env_vars

    set_env_vars(
        jax_debug_nans=config.debug_nans,
        cuda_visible_devices=config.cuda_visible_devices,
    )

    import os
    import wandb
    from omegaconf import OmegaConf

    import jax
    import jax.numpy as jnp
    import optax

    from stable_baselines3.common.callbacks import EventCallback

    from diff_trans.envs.gym_wrapper import get_env, BaseEnv
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions, evaluate_policy
    from diff_trans.utils.callbacks import (
        StopTrainingOnRewardThreshold,
        EvalCallback,
        SaveBestModelCallback,
        MultiCallback,
    )

    from constants import ALGORITHMS
    from utils import default

    Algorithm = ALGORITHMS[config.algorithm]

    def create_env(
        Env: Type[BaseEnv],
        parameter: Optional[jnp.ndarray] = None,
        precompile: bool = False,
        eval_precompile: bool = False,
    ) -> Tuple[BaseEnv, BaseEnv]:
        env = Env(
            num_envs=config.train.num_envs,
            **default(config.env_config, {}),
            precompile=precompile,
        )
        eval_env = Env(
            num_envs=config.eval.num_episodes,
            **default(config.env_config, {}),
            precompile=eval_precompile,
        )

        if parameter is not None:
            env.set_model_parameter(parameter)
            eval_env.set_model_parameter(parameter)

        return env, eval_env

    # Get default parameter and parameter range
    Env = get_env(config.env_name)
    sim_env, sim_eval_env = create_env(Env)

    default_parameter = sim_env.get_model_parameter()
    num_parameters = default_parameter.shape[0]

    # Determine parameters to adapt and their values
    param_ids = jnp.array(default(config.param_ids, []), dtype=int)
    param_values = jnp.array(default(config.param_values, []), dtype=float)

    # Setup envs and parameters
    target_parameter = default_parameter.at[param_ids].set(param_values)
    preal_env, preal_eval_env = create_env(Env, parameter=target_parameter)

    if config.freeze_all:
        parameter_mask = param_ids
    elif config.freeze_ids is not None:
        parameter_mask = jnp.array(
            [i for i in range(num_parameters) if i not in config.freeze_ids],
            dtype=int,
        )
    else:
        parameter_mask = jnp.arange(0, num_parameters, dtype=int)

    class TuneCallback(EventCallback):
        def __init__(
            self,
            sim_env: BaseEnv,
            preal_env: BaseEnv,
            update_env: Callable[[jnp.ndarray], None],
            parameter: jnp.ndarray,
            parameter_range: jnp.ndarray,
            parameter_mask: jnp.ndarray,
            loss_function: Callable[[BaseDiffEnv, jnp.ndarray, List], jnp.ndarray],
            tune_config: TuneConfig,
            log_wandb: bool = True,
            verbose: int = 1,
        ):
            super().__init__(verbose=verbose)

            self.sim_env = sim_env
            self.preal_env = preal_env
            self.update_env = update_env

            self.parameter = parameter
            self.parameter_range = parameter_range
            self.parameter_mask = parameter_mask
            self.masked_parameter = parameter[parameter_mask]

            self.optimizer = optax.adam(tune_config.learning_rate)
            self.optimizer_state = self.optimizer.init(self.masked_parameter)

            self.loss_function = loss_function
            self.grad_function = jax.grad(self.loss_function, argnums=1)

            self.tune_config = tune_config

            self.num_gradient_steps = 0
            self.current_loss = float("inf")

            self.num_rollouts = 0
            self.rollouts = []

            self.log_wandb = log_wandb
            self.verbose = verbose

        def rollout(self):
            self.num_rollouts += 1

            if self.verbose >= 1:
                print(
                    f"Collecting rollouts ({self.num_rollouts}/{self.tune_config.max_rollouts})"
                )

            rollouts = rollout_transitions(
                self.preal_env,
                self.model,
                num_transitions=self.tune_config.rollout_timesteps,
            )
            self.rollouts.append(rollouts)

        def get_loss(self):
            loss = 0
            for rollouts in self.rollouts:
                loss += self.loss_function(
                    self.sim_env.diff_env, self.parameter, rollouts
                )

            return loss / len(self.rollouts)

        def get_grad(self):
            grad = 0
            for rollouts in self.rollouts:
                grad += self.grad_function(
                    self.sim_env.diff_env, self.parameter, rollouts
                )

            return grad / len(self.rollouts)

        def tune(self):
            print(f"Tuning parameter...")

            for _ in range(self.tune_config.gradient_steps):
                self.num_gradient_steps += 1

                grad = self.get_grad()
                masked_grad = grad[self.parameter_mask]

                updates, self.optimizer_state = self.optimizer.update(
                    masked_grad, self.optimizer_state
                )
                self.masked_parameter = optax.apply_updates(
                    self.masked_parameter, updates
                )
                self.parameter = self.parameter.at[self.parameter_mask].set(
                    self.masked_parameter
                )
                self.parameter = jnp.clip(
                    self.parameter, self.parameter_range[0], self.parameter_range[1]
                )

                self.update_env(self.parameter)

                loss = self.get_loss()
                self.current_loss = loss.tolist()

                if self.verbose >= 1:
                    print(f"  Steps: {self.num_gradient_steps}")
                    print(f"    Loss: {self.current_loss:.6f}")
                if self.verbose >= 2:
                    print(f"    Grad: {grad.tolist()}")
                    print(f"    Parameter: {self.parameter.tolist()}")

                if self.log_wandb:
                    wandb.log(
                        dict(
                            gradient_steps=self.num_gradient_steps,
                            loss=loss,
                        )
                    )

        def _on_step(self) -> bool:
            if self.n_calls < self.tune_config.start_rollout_at:
                return True
            if self.current_loss < self.tune_config.stop_tuning_below:
                return True

            if (
                self.n_calls % self.tune_config.rollout_frequency == 0
                and self.num_rollouts < self.tune_config.max_rollouts
            ):
                self.rollout()

            if self.n_calls % self.tune_config.tune_frequency == 0:
                self.tune()

            return True

    for exp_id in range(config.num_exp):
        # Create run
        if config.log_wandb:
            run_name = "-".join([*exp_levels, name, f"{exp_id:02d}"])
            tags = [
                *exp_levels,
                name,
                f"{exp_id:02d}",
                config.env_name,
                config.algorithm,
            ]
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                tags=tags,
                config=OmegaConf.to_container(config),
            )

        try:
            print(f"Target parameter: {target_parameter}")
            print(f"Default parameter: {default_parameter}")
            print()

            # Start parameter tuning
            print(f"Experiment {exp_id}")
            print()

            model_name = f"{config.algorithm}-{config.env_name}-{exp_id:02d}"
            model_path = os.path.join(models_dir, f"{model_name}.zip")

            sim_gym_env = sim_env.create_gym_env(**default(config.env_config, {}))
            model = Algorithm(
                "MlpPolicy",
                sim_gym_env,
                verbose=0,
                **default(config.algorithm_config, {}),
            )

            print("Model path:", model_path)
            if config.train.use_checkpoint and os.path.exists(model_path):
                print("Loading model from", model_path)
                model = model.load(model_path, env=sim_gym_env)

            def update_env(parameter: jnp.ndarray):
                sim_env.set_model_parameter(parameter)
                sim_eval_env.set_model_parameter(parameter)

            tune_callback = TuneCallback(
                sim_env,
                preal_env,
                update_env,
                default_parameter.copy(),
                sim_env.diff_env.parameter_range,
                parameter_mask,
                single_transition_loss,
                config.tune,
                log_wandb=config.log_wandb,
                verbose=config.verbose,
            )

            save_best_model_callback = SaveBestModelCallback(
                model_path=model_path, verbose=config.verbose
            )

            callback_on_log = (
                (lambda metrics: wandb.log(metrics)) if config.log_wandb else None
            )
            sim_eval_callback = EvalCallback(
                sim_eval_env,
                prefix="sim_",
                n_eval_episodes=config.eval.num_episodes,
                callback_on_new_best=save_best_model_callback,
                callback_on_log=callback_on_log,
                eval_freq=config.eval.frequency // config.train.num_envs,
                verbose=config.verbose,
            )
            preal_eval_callback = EvalCallback(
                preal_eval_env,
                prefix="preal_",
                n_eval_episodes=config.eval.num_episodes,
                callback_on_new_best=save_best_model_callback,
                callback_on_log=callback_on_log,
                eval_freq=config.eval.frequency // config.train.num_envs,
                verbose=config.verbose,
            )

            model.learn(
                total_timesteps=config.train.max_timesteps,
                callback=[sim_eval_callback, preal_eval_callback, tune_callback],
                progress_bar=True,
            )

            if config.log_wandb:
                wandb.finish()

        except Exception as e:
            if config.log_wandb:
                wandb.finish(exit_code=1)
            raise e


if __name__ == "__main__":
    app()
