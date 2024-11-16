from typing import List, Optional, Dict, Any, Type, Tuple

import typer
from experiments.utils.config import *

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class TrainConfig:
    num_envs: int = 16
    max_timesteps: int = 50000
    threshold: float = 150
    load_last_model: bool = False
    use_checkpoint: bool = False


@dataclass
class TuneConfig:
    learning_rate: float = 1e-2
    gradient_steps: int = 1
    epochs: int = 10
    loss_rollout_length: int = 1000


@dataclass
class EvalConfig:
    num_episodes: int = 256
    frequency: int = 10000


# Configuration dataclass for experiment settings
@dataclass
class Config:
    cuda_visible_devices: Optional[List[str]] = None
    parallel_instances: Optional[int] = None
    start_exp: int = 0
    num_exp: int = 3

    algorithm: str = "PPO"
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
    import datetime
    from experiments.utils.exp import load_config

    config, exp_levels, models_dir = load_config(
        __file__, name, Config, config_path=config_path
    )
    if config is None:
        print("Configuration created")
        return

    # Distribute experiments in tmux
    if config.num_exp > 0 and config.parallel_instances is not None:
        # import uuid
        from experiments.utils.path import get_exp_module_name
        from experiments.utils.tmux import create_grid_window

        from omegaconf import OmegaConf

        os.makedirs("parallel_tmp", exist_ok=True)

        module_name = get_exp_module_name("experiments", __file__)
        commands = ["python", "-m", module_name, name]

        num_exp_per_parallel = config.num_exp // config.parallel_instances
        num_exps = [num_exp_per_parallel for _ in range(config.parallel_instances)]
        for i in range(
            config.num_exp - num_exp_per_parallel * config.parallel_instances
        ):
            num_exps[i] += 1

        window = create_grid_window(name, config.parallel_instances)
        config.parallel_instances = None
        start_exp = config.start_exp

        for i, num_exp in enumerate(num_exps):
            # config_id = uuid.uuid4().hex
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            tmp_config_path = os.path.join(
                "parallel_tmp", f"{name}-{i}-{timestamp}.json"
            )
            tmp_config_file = open(tmp_config_path, "w")

            config.start_exp = start_exp
            config.num_exp = num_exp
            start_exp = start_exp + num_exp
            OmegaConf.save(config, tmp_config_file)

            pane = window.panes[i]
            pane_commands = commands.copy()
            pane_commands.append(f"--config-path={tmp_config_path}")
            pane.send_keys(" ".join(pane_commands))

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

    from diff_trans.envs.gym_wrapper import get_env, BaseEnv
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_trajectories, evaluate_policy
    from diff_trans.utils.callbacks import (
        StopTrainingOnRewardThreshold,
        EvalCallback,
        SaveBestModelCallback,
        MultiCallback,
    )

    from constants import ALGORITHMS
    from utils import default

    Algorithm = ALGORITHMS[config.algorithm]

    exp_start = config.start_exp
    exp_end = exp_start + 1
    if config.num_exp > 0:
        exp_end = exp_start + config.num_exp

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
    sim_diff_env = sim_env.diff_env

    default_parameter = sim_env.get_model_parameter()
    num_parameters = default_parameter.shape[0]
    parameter_min, parameter_max = sim_diff_env.parameter_range

    # Determine parameters to adapt and their values
    param_ids = jnp.array(default(config.param_ids, []), dtype=int)
    param_values = jnp.array(default(config.param_values, []), dtype=float)

    # Setup envs and parameters
    target_parameter = default_parameter.at[param_ids].set(param_values)
    preal_env, preal_eval_env = create_env(Env, parameter=target_parameter)

    if config.freeze_all:
        param_selections = param_ids
    elif config.freeze_ids is not None:
        param_selections = jnp.array(
            [i for i in range(num_parameters) if i not in config.freeze_ids],
            dtype=int,
        )
    else:
        param_selections = jnp.arange(0, num_parameters, dtype=int)

    for exp_id in range(exp_start, exp_end):
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

            def loss_of_single_param(parameter, rollouts):
                return single_transition_loss(sim_diff_env, parameter, rollouts)

            compute_loss_g = jax.grad(loss_of_single_param, argnums=0)

            # Start parameter tuning
            print(f"Experiment {exp_id}")
            print()

            parameter = default_parameter.copy()
            parameter_subset = parameter[param_selections]
            optimizer = optax.adam(config.tune.learning_rate)
            opt_state = optimizer.init(parameter_subset)
            preal_steps = 0
            last_model_path = None

            rollouts = []

            for i in range(config.tune.epochs):
                print(f"Iteration {i}")
                print()

                model_name = (
                    f"{config.algorithm}-{config.env_name}-{exp_id:02d}-{i:02d}"
                )
                model_path = os.path.join(models_dir, f"{model_name}.zip")

                sim_gym_env = sim_env.create_gym_env(**default(config.env_config, {}))
                model = Algorithm(
                    "MlpPolicy",
                    sim_gym_env,
                    verbose=0,
                    **default(config.algorithm_config, {}),
                )

                print("Model path:", model_path)
                if config.train.load_last_model and last_model_path is not None:
                    print("Loading model from", last_model_path)
                    model = model.load(last_model_path, env=sim_gym_env)
                elif config.train.use_checkpoint and os.path.exists(model_path):
                    print("Loading model from", model_path)
                    model = model.load(model_path, env=sim_gym_env)

                save_best_model_callback = SaveBestModelCallback(
                    model_path=model_path, verbose=1
                )
                callback_on_best = StopTrainingOnRewardThreshold(
                    reward_threshold=config.train.threshold, verbose=1
                )
                eval_callback = EvalCallback(
                    sim_eval_env,
                    n_eval_episodes=config.eval.num_episodes,
                    callback_on_new_best=MultiCallback(
                        [save_best_model_callback, callback_on_best]
                    ),
                    eval_freq=config.eval.frequency // sim_env.num_envs,
                    verbose=1,
                )

                model.learn(
                    total_timesteps=config.train.max_timesteps,
                    callback=eval_callback,
                    progress_bar=True,
                )

                last_model_path = model_path
                model = model.load(model_path, env=sim_gym_env)

                # Evaluate model
                print("Evaluating sim model")
                sim_eval = evaluate_policy(
                    sim_eval_env,
                    model,
                    config.eval.num_episodes,
                    progress_bar=True,
                )
                print(f"Sim performance: {sim_eval[0]:6.2f} +/- {sim_eval[1]:6.2f}\n")

                print("Evaluating preal model")
                preal_eval = evaluate_policy(
                    preal_eval_env,
                    model,
                    config.eval.num_episodes,
                    progress_bar=True,
                )
                print(
                    f"Preal performance: {preal_eval[0]:6.2f} +/- {preal_eval[1]:6.2f}\n"
                )

                print("Computing gradient")
                rollouts.append(
                    rollout_trajectories(
                        preal_env,
                        model,
                        num_transitions=config.tune.loss_rollout_length,
                    )
                )
                preal_steps += config.tune.loss_rollout_length

                for gi in range(config.tune.gradient_steps):
                    print(f"  Gradient step {gi}")

                    loss = 0
                    grad = 0
                    for transitions in rollouts:
                        loss += loss_of_single_param(parameter, transitions)
                        grad += compute_loss_g(parameter, transitions)

                    loss /= len(rollouts)
                    grad /= len(rollouts)

                    if jnp.isnan(grad).any():
                        print("Nan in grad\n")
                        raise ValueError("Nan in grad")

                    print(f"    Parameter = {parameter_subset}")
                    print(f"    Loss = {loss}")
                    print(f"    Grad = {grad}")

                    wandb.log(
                        dict(
                            gradient_steps=i * config.tune.gradient_steps + gi,
                            loss=loss,
                        )
                    )

                    grad_subset = grad[param_selections]
                    updates, opt_state = optimizer.update(grad_subset, opt_state)
                    parameter_subset = optax.apply_updates(parameter_subset, updates)
                    parameter = parameter.at[param_selections].set(parameter_subset)
                    # Ensure parameter is within bounds
                    parameter = jnp.clip(parameter, parameter_min, parameter_max)

                if config.log_wandb:
                    param_metrics = dict(
                        (f"param_{i}", parameter[i]) for i in param_ids
                    )
                    metrics = dict(
                        iteration=i,
                        preal_steps=preal_steps,
                        sim_eval_mean=sim_eval[0],
                        sim_eval_std=sim_eval[1],
                        preal_eval_mean=preal_eval[0],
                        preal_eval_std=preal_eval[1],
                        loss=loss,
                        **param_metrics,
                    )
                    wandb.log(metrics)

                print(f"Sim parameter: {parameter}")
                print(f"Preal parameter: {target_parameter}")
                print()

                sim_env.set_model_parameter(parameter)
                sim_env.update_gym_env(sim_gym_env, parameter)
                sim_eval_env.set_model_parameter(parameter)

                del model

            if config.log_wandb:
                wandb.finish()

        except Exception as e:
            if config.log_wandb:
                wandb.finish(exit_code=1)
            raise e


if __name__ == "__main__":
    app()
