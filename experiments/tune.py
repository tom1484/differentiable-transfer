from typing import List, Optional, Dict, Any, Type, Tuple, Callable

import typer
from diff_trans.envs.base import BaseDiffEnv
from experiments.utils.config import *

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class TrainConfig:
    num_envs: int = 1
    max_timesteps: int = 100000
    checkpoint: Optional[str] = None


@dataclass
class TuneConfig:
    stop_tuning_below: float = 1.0
    tune_frequency: int = 1000

    batch_size: int = 100
    learning_rate: float = 1e-2
    gradient_steps: int = 1

    start_rollout_at: int = 10000
    max_rollouts: int = 10
    rollout_frequency: int = 5000
    rollout_timesteps: int = 500

    skip_nan: bool = False
    clip_gradient: float = 50
    gradient_threshold: float = 100


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

    import wandb
    from omegaconf import OmegaConf

    import os
    import time
    import pickle
    import traceback

    import jax
    import jax.numpy as jnp
    import optax

    from stable_baselines3.common.callbacks import EventCallback

    from diff_trans.envs.gym_wrapper import get_env, BaseEnv
    from diff_trans.utils.loss import (
        single_transition_loss,
        extract_array_from_transitions,
    )
    from diff_trans.utils.rollout import rollout_trajectories
    from diff_trans.utils.callbacks import EvalCallback

    from constants import ALGORITHMS
    from utils import default

    Algorithm = ALGORITHMS[config.algorithm]

    def create_env(
        Env: Type[BaseEnv],
        parameter: Optional[jax.Array] = None,
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

    loss_env = Env(
        num_envs=config.tune.batch_size,
        **default(config.env_config, {}),
    )

    class TuneCallback(EventCallback):
        def __init__(
            self,
            env: BaseEnv,
            rollout_env: BaseEnv,
            update_env: Callable[[jax.Array], None],
            parameter: jax.Array,
            parameter_range: jax.Array,
            parameter_mask: jax.Array,
            loss_function: Callable[
                [BaseDiffEnv, jax.Array, jax.Array, jax.Array, jax.Array],
                jax.Array,
            ],
            config: TuneConfig,
            log_wandb: bool = True,
            verbose: int = 1,
        ):
            super().__init__(verbose=verbose)

            self.env = env
            self.rollout_env = rollout_env
            self.update_env = update_env

            self.parameter = parameter
            self.parameter_range = parameter_range
            self.parameter_mask = parameter_mask
            self.masked_parameter = parameter[parameter_mask]

            self.optimizer = optax.adam(config.learning_rate)
            self.optimizer_state = self.optimizer.init(self.masked_parameter)

            self.create_loss_function(loss_function, config.batch_size)
            self.num_gradient_steps = 0
            self.current_loss = float("inf")

            self.tune_config = config

            self.num_rollouts = 0
            self.num_transitions = 0
            self.transitions = (jnp.empty((0,)), jnp.empty((0,)), jnp.empty((0,)))

            self.log_wandb = log_wandb
            self.verbose = verbose

            self.noise_scale = 0.001

        def create_loss_function(
            self,
            loss_function: Callable[
                [BaseDiffEnv, jax.Array, jax.Array, jax.Array, jax.Array],
                jax.Array,
            ],
            batch_size: int,
        ):
            parameter = self.parameter.copy()
            state_dim = self.env.diff_env.state_dim
            control_dim = self.env.diff_env.control_dim

            states = jnp.zeros((batch_size, state_dim))
            next_states = jnp.zeros((batch_size, state_dim))
            actions = jnp.zeros((batch_size, control_dim))

            loss_function_jit = jax.jit(loss_function, static_argnums=(0,))
            grad_function_jit = jax.jit(
                jax.grad(loss_function, argnums=1), static_argnums=(0,)
            )

            print("Compiling loss function... ", end="")
            self.loss_function = loss_function_jit.lower(
                self.env.diff_env, parameter, states, next_states, actions
            ).compile()
            self.grad_function = grad_function_jit.lower(
                self.env.diff_env, parameter, states, next_states, actions
            ).compile()
            print("Done")

            # self.loss_function = loss_function
            # self.grad_function = jax.grad(loss_function, argnums=1)

            # self.loss_function = jax.jit(loss_function, static_argnums=(0,))
            # self.grad_function = jax.jit(
            #     jax.grad(loss_function, argnums=1), static_argnums=(0,)
            # )

        def rollout(self):
            self.num_rollouts += 1

            if self.verbose >= 1:
                print(
                    f"Collecting rollouts ({self.num_rollouts}/{self.tune_config.max_rollouts})"
                )

            trajectories = rollout_trajectories(
                self.rollout_env,
                self.model,
                num_transitions=self.tune_config.rollout_timesteps,
            )
            transitions = []
            for trajectory in trajectories:
                transitions.extend(trajectory)

            states, next_states, actions = extract_array_from_transitions(transitions)

            if self.num_transitions == 0:
                self.transitions = (states, next_states, actions)
                self.num_transitions = len(states)
            else:
                self.transitions = (
                    jnp.concatenate([self.transitions[0], states]),
                    jnp.concatenate([self.transitions[1], next_states]),
                    jnp.concatenate([self.transitions[2], actions]),
                )
                self.num_transitions += len(states)

            if self.log_wandb:
                wandb.log(
                    dict(
                        timesteps=self.num_timesteps,
                        num_rollouts=self.num_rollouts,
                        num_transitions=self.num_transitions,
                    )
                )

        def shuffle_transitions(self):
            states, next_states, actions = self.transitions
            key = jax.random.PRNGKey(time.time_ns())
            self.transitions = (
                jax.random.permutation(key, states, axis=0, independent=True),
                jax.random.permutation(key, next_states, axis=0, independent=True),
                jax.random.permutation(key, actions, axis=0, independent=True),
            )

        def sample_transitions_compute(
            self,
            compute_function: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
            default_value: Optional[jax.Array] = None,
            # clip_value: Optional[jax.Array] = None,
            # ignore_threshold: Optional[jax.Array] = None,
            clip_value: Optional[float] = None,
            ignore_threshold: Optional[float] = None,
        ) -> jax.Array:
            batch_size = self.tune_config.batch_size
            num_batches = self.num_transitions // batch_size

            value_acc = 0
            success = 0
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size

                states = self.transitions[0][start_index:end_index]
                next_states = self.transitions[1][start_index:end_index]
                actions = self.transitions[2][start_index:end_index]

                value = compute_function(states, next_states, actions)
                if jnp.isnan(value).any():
                    if self.tune_config.skip_nan:
                        continue
                    else:
                        traceback.print_stack()
                        raise ValueError("Encountered NaN")

                if ignore_threshold is not None and jnp.any(
                    jnp.abs(value) > ignore_threshold
                ):
                    continue

                if clip_value is not None:
                    value = jnp.clip(value, -clip_value, clip_value)

                value_acc += value
                success += 1

            if success == 0:
                if default_value is None:
                    traceback.print_stack()
                    raise ValueError("No valid transitions found")
                else:
                    return default_value

            return value_acc / success

        def compute_loss(self, parameter: jax.Array):
            return self.sample_transitions_compute(
                lambda states, next_states, actions: self.loss_function(
                    parameter,
                    states,
                    next_states,
                    actions,
                    # self.env.diff_env, parameter, states, next_states, actions
                ),
                default_value=jnp.zeros(1),
            )

        def compute_grad(self, parameter: jax.Array):
            return self.sample_transitions_compute(
                lambda states, next_states, actions: self.grad_function(
                    parameter,
                    states,
                    next_states,
                    actions,
                    # self.env.diff_env, parameter, states, next_states, actions
                ),
                default_value=jnp.zeros(parameter.shape),
                clip_value=config.tune.clip_gradient,
                ignore_threshold=config.tune.gradient_threshold,
            )

        def tune(self):
            if self.num_transitions < self.tune_config.batch_size:
                return

            print(f"Tuning parameter...")
            for _ in range(self.tune_config.gradient_steps):
                self.num_gradient_steps += 1
                self.shuffle_transitions()

                grad = self.compute_grad(self.parameter)
                masked_grad = grad[self.parameter_mask]

                updates, self.optimizer_state = self.optimizer.update(
                    masked_grad, self.optimizer_state
                )
                masked_parameter = optax.apply_updates(self.masked_parameter, updates)
                parameter = self.parameter.at[self.parameter_mask].set(masked_parameter)
                parameter = jnp.clip(
                    parameter, self.parameter_range[0], self.parameter_range[1]
                )

                loss = self.compute_loss(parameter)

                if self.verbose >= 1:
                    print(f"  Steps: {self.num_gradient_steps}")
                    print(f"    Loss: {loss:.6f}")
                if self.verbose >= 2:
                    print(f"    Grad: {grad.tolist()}")
                    print(f"    Parameter: {parameter.tolist()}")

                if jnp.isnan(loss) or jnp.isnan(grad).any():
                    print("Loss or grad is NaN, skipping update")

                    # Save parameter and rollouts to file
                    filename = "nan-param-rollouts.pkl"
                    if not os.path.exists(os.path.join(models_dir, filename)):
                        with open(os.path.join(models_dir, filename), "wb") as f:
                            pickle.dump(
                                dict(
                                    # env=self.env,
                                    parameter=self.parameter,
                                    transitions=self.transitions,
                                ),
                                f,
                            )

                    breakpoint()

                    key = jax.random.PRNGKey(time.time_ns())
                    noise = jax.random.normal(key=key, shape=masked_parameter.shape)
                    masked_parameter = self.masked_parameter + noise * self.noise_scale
                    parameter = self.parameter.at[self.parameter_mask].set(
                        masked_parameter
                    )

                    self.parameter = parameter
                    self.masked_parameter = masked_parameter

                    continue

                if jnp.any(jnp.abs(grad) > 100.0):
                    # Save parameter and rollouts to file
                    filename = "large-grad-param-rollouts.pkl"
                    if not os.path.exists(os.path.join(models_dir, filename)):
                        with open(os.path.join(models_dir, filename), "wb") as f:
                            pickle.dump(
                                dict(
                                    # env=self.env,
                                    parameter=self.parameter,
                                    transitions=self.transitions,
                                ),
                                f,
                            )

                    grad = grad.at[:].set(0.0)

                filename = f"loss-{loss:.6f}-param-rollouts.pkl"
                if not os.path.exists(os.path.join(models_dir, filename)):
                    with open(os.path.join(models_dir, filename), "wb") as f:
                        pickle.dump(
                            dict(
                                # env=self.env,
                                parameter=self.parameter,
                                transitions=self.transitions,
                            ),
                            f,
                        )

                self.current_loss = loss.tolist()
                self.update_env(parameter)

                self.parameter = parameter
                self.masked_parameter = masked_parameter

                if self.log_wandb:
                    wandb.log(
                        dict(
                            gradient_steps=self.num_gradient_steps,
                            loss=loss,
                        )
                    )

        def _on_step(self) -> bool:
            if (
                self.n_calls < self.tune_config.start_rollout_at
                or self.current_loss < self.tune_config.stop_tuning_below
            ):
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

            def update_env(parameter: jax.Array):
                sim_env.set_model_parameter(parameter)
                sim_env.update_gym_env(sim_gym_env, parameter)
                sim_eval_env.set_model_parameter(parameter)

            tune_callback = TuneCallback(
                env=loss_env,
                rollout_env=preal_env,
                update_env=update_env,
                parameter=default_parameter.copy(),
                parameter_range=sim_env.diff_env.parameter_range,
                parameter_mask=parameter_mask,
                loss_function=single_transition_loss,
                config=config.tune,
                log_wandb=config.log_wandb,
                verbose=config.verbose,
            )

            callback_on_log = (
                (lambda metrics: wandb.log(metrics)) if config.log_wandb else None
            )
            sim_eval_callback = EvalCallback(
                sim_eval_env,
                prefix="sim_",
                n_eval_episodes=config.eval.num_episodes,
                callback_on_log=callback_on_log,
                eval_freq=config.eval.frequency // config.train.num_envs,
                verbose=config.verbose,
            )
            preal_eval_callback = EvalCallback(
                preal_eval_env,
                prefix="preal_",
                n_eval_episodes=config.eval.num_episodes,
                callback_on_log=callback_on_log,
                eval_freq=config.eval.frequency // config.train.num_envs,
                verbose=config.verbose,
            )

            model_name = f"{config.algorithm}-{config.env_name}-{exp_id:02d}"
            model_path = os.path.join(models_dir, f"{model_name}.zip")

            sim_gym_env = sim_env.create_gym_env(**default(config.env_config, {}))
            model = Algorithm(
                "MlpPolicy",
                sim_gym_env,
                verbose=0,
                **default(config.algorithm_config, {}),
            )

            if config.train.checkpoint is not None:
                print("Loading model from", config.train.checkpoint)
                model = model.load(config.train.checkpoint, env=sim_gym_env)
            else:
                print("No checkpoint provided, training from scratch")
                print("Model path:", model_path)

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
