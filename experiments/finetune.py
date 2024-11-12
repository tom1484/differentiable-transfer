import typer
from typing import List, Union, cast, Optional, Dict, Any
from dataclasses import dataclass
from dataclasses_json import dataclass_json

app = typer.Typer(pretty_exceptions_show_locals=False)


# Configuration dataclass for experiment settings
@dataclass_json
@dataclass
class CONFIG:
    cuda_visible_devices: Optional[List[str]] = None
    exp_start: int = 0
    num_exp: int = 3

    algorithm: str = "PPO"
    algorithm_config: Optional[Dict[str, Any]] = None
    env_name: str = "InvertedPendulum-v5"

    baseline_max_timesteps: int = 1000000
    baseline_threshold: float = 100
    baseline_num_envs: int = 256
    baseline_eval_frequency: int = 10000

    adapt_params: Union[None, int, List[int]] = None
    param_values: Union[None, float, List[float]] = None
    param_deviations: Union[float, List[float]] = 0.3

    adapt_max_timesteps: int = 1000000
    adapt_threshold: int = 100
    adapt_num_envs: int = 256

    eval_num_episodes: int = 256
    eval_frequency: int = 10000

    log_wandb: bool = True
    override: bool = False
    debug_nans: bool = False


# Main entry point for the experiment
@app.command()
def main(name: str = typer.Argument(..., help="Name of the experiment")):
    from experiments.utils.exp import load_config
    from experiments.env import set_env_vars

    config, exp_levels, models_dir = load_config(__file__, name, CONFIG)
    if config is None:
        print("Configuration created")
        return

    # Set up JAX configuration
    set_env_vars(
        jax_debug_nans=config.debug_nans,
        cuda_visible_devices=config.cuda_visible_devices,
    )

    import os
    import wandb

    from jax import numpy as jnp

    from diff_trans.envs.gym_wrapper import get_env
    from diff_trans.utils.rollout import evaluate_policy

    from diff_trans.utils.callbacks import (
        StopTrainingOnRewardThreshold,
        EvalCallback,
    )

    from experiments.utils.exp import convert_arg_array
    from constants import ALGORITHMS

    Algorithm = ALGORITHMS[config.algorithm]

    # Initialize environments and parameters
    print(f"Initializing environment {config.env_name}")
    Env = get_env(config.env_name)
    sim_env = Env(num_envs=config.baseline_num_envs)
    preal_env = Env(num_envs=config.adapt_num_envs)
    print(f"Environment initialized")

    sim_env_conf = sim_env.diff_env
    preal_env_conf = preal_env.diff_env

    # Get default parameter and parameter range
    default_parameter = preal_env_conf.get_parameter()
    num_parameters = default_parameter.shape[0]
    parameter_range = preal_env_conf.parameter_range
    _, parameter_max = parameter_range

    # Determine parameters to adapt and their values
    if config.adapt_params is None:
        adapt_param_ids = jnp.arange(0, num_parameters)
    else:
        adapt_param_ids = convert_arg_array(config.adapt_params, int)

    num_adapt_parameters = adapt_param_ids.shape[0]
    if config.param_values is not None:
        values = convert_arg_array(config.param_values, float, num_adapt_parameters)
    else:
        deviations = convert_arg_array(
            config.param_deviations, float, num_adapt_parameters
        )
        values = (
            default_parameter[adapt_param_ids]
            + deviations * (parameter_max - default_parameter)[adapt_param_ids]
        )

    target_parameter = default_parameter.at[adapt_param_ids].set(values)
    preal_env_conf.model = preal_env_conf.set_parameter(target_parameter)

    # Set up evaluation environments
    sim_eval_env = Env(num_envs=config.eval_num_episodes)
    preal_eval_env = Env(num_envs=config.eval_num_episodes)

    sim_eval_env.diff_env.model = sim_env_conf.model
    preal_eval_env.diff_env.model = preal_env_conf.model

    print(f"Target parameter: {target_parameter}")
    print(f"Default parameter: {default_parameter}\n")

    # Train baseline model
    model_name = f"{config.algorithm}-{config.env_name}-baseline"
    model = Algorithm("MlpPolicy", sim_env, verbose=1, **config.algorithm_config)
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if os.path.exists(model_path) and not config.override:
        print(f"Loading baseline model from {model_path}")
        model = model.load(model_path, sim_env)

        baseline_eval = evaluate_policy(sim_eval_env, model, config.eval_num_episodes)
    else:
        # Initialize wandb logging for baseline training
        if config.log_wandb:
            run_name = "-".join([*exp_levels, name, "baseline"])
            tags = [*exp_levels, name, "baseline", config.env_name, config.algorithm]
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                tags=tags,
                config={
                    "env_name": config.env_name,
                    "baseline_max_timesteps": config.baseline_max_timesteps,
                    "baseline_num_envs": config.baseline_num_envs,
                    "param_deviations": config.param_deviations,
                    "eval_num_episodes": config.eval_num_episodes,
                    "eval_frequency": config.eval_frequency,
                },
            )

        def callback_on_log(metrics):
            if config.log_wandb:
                wandb.log(metrics)

        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=config.baseline_threshold, verbose=1
        )
        eval_callback = EvalCallback(
            sim_eval_env,
            n_eval_episodes=config.eval_num_episodes,
            callback_on_new_best=callback_on_best,
            # callback_on_log=callback_on_log if config.log_wandb else None,
            callback_on_log=callback_on_log,
            eval_freq=config.baseline_eval_frequency // sim_env.num_envs,
            verbose=0,
        )

        # Train baseline model until performance threshold is reached
        model.learn(
            total_timesteps=config.baseline_max_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
        baseline_eval = evaluate_policy(sim_eval_env, model, config.eval_num_episodes)
        print(f"Baseline evaluation: {baseline_eval}\n")

        model.save(model_path)
        if config.log_wandb:
            wandb.finish()

    # Start adaptation experiments
    for i in range(config.exp_start, config.exp_start + config.num_exp):
        try:
            # Initialize wandb logging for adaptation experiment
            if config.log_wandb:
                run_name = "-".join([*exp_levels, name, f"{i:02d}"])
                tags = [
                    *exp_levels,
                    name,
                    f"{i:02d}",
                    config.env_name,
                    config.algorithm,
                ]
                wandb.init(
                    project="differentiable-transfer",
                    name=run_name,
                    tags=tags,
                    config={
                        "id": i,
                        "env_name": config.env_name,
                        "baseline_mean_return": baseline_eval[0],
                        "param_deviations": config.param_deviations,
                        "adapt_threshold": config.adapt_threshold,
                        "adapt_num_envs": config.adapt_num_envs,
                        "eval_frequency": config.eval_frequency,
                    },
                )

            def callback_on_log(metrics):
                if config.log_wandb:
                    wandb.log(metrics)
                # else:
                #     print(metrics)

            callback_on_best = StopTrainingOnRewardThreshold(
                reward_threshold=config.adapt_threshold, verbose=1
            )
            eval_callback = EvalCallback(
                preal_eval_env,
                n_eval_episodes=config.eval_num_episodes,
                callback_on_new_best=callback_on_best,
                callback_on_log=callback_on_log,
                eval_freq=config.eval_frequency // preal_env.num_envs,
                verbose=0,
            )

            print(f"Running experiment {i}")
            print()

            # Load baseline model and adapt to new environment
            model = Algorithm(
                "MlpPolicy", preal_env, verbose=1, **config.algorithm_config
            )
            model = model.load(model_path, preal_env)

            # Evaluate initial performance
            eval_stats = evaluate_policy(
                preal_eval_env, model, config.eval_num_episodes
            )
            print(f"Initial evaluation stats: {eval_stats}")

            # Adapt model to new environment
            model = model.learn(
                total_timesteps=config.adapt_max_timesteps,
                callback=eval_callback,
                progress_bar=True,
            )
            # Evaluate final performance
            eval_stats = evaluate_policy(
                preal_eval_env, model, n_eval_episodes=config.eval_num_episodes
            )
            print(f"Final evaluation stats: {eval_stats}")

            if config.log_wandb:
                wandb.finish()

            del model

        except Exception as e:
            if config.log_wandb:
                wandb.finish(exit_code=1)
            raise e


if __name__ == "__main__":
    app()
