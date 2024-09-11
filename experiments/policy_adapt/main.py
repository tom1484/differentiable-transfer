import typer
from typing import List, Union, cast, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

app = typer.Typer(pretty_exceptions_show_locals=False)

# Configuration dataclass for experiment settings
@dataclass_json
@dataclass
class CONFIG:
    name: str
    gpu: Optional[int] = None
    exp_start: int = 0
    num_exp: int = 3
    env_name: str = "InvertedPendulum-v1"
    baseline_round_timesteps: int = 5e4
    baseline_threshold: float = 150
    baseline_num_envs: int = 16
    adapt_params: Union[None, int, List[int]] = None
    param_values: Union[None, float, List[float]] = None
    param_deviations: Union[float, List[float]] = 0.3
    adapt_train_lr: float = 1e-2
    adapt_timesteps: int = 5e5
    adapt_num_envs: int = 16
    log_wandb: bool = True
    eval_num_steps: int = 1e4
    eval_num_episodes: int = 256
    override: bool = False
    debug_nans: bool = False

# Main entry point for the experiment
@app.command()
def main(name: str = typer.Argument(..., help="Name of the experiment")):
    import json
    from definitions import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    # Initialize default configuration
    default_config = CONFIG(name=name)

    # Create experiment assets (folders and default configuration)
    exp_levels = get_exp_file_levels("experiments", __file__)
    new_config, config_path, models_dir = create_exp_assets(
        ROOT_DIR, exp_levels, name, default_config.to_dict()
    )

    if new_config:
        print("Configuration created")
        return

    # Load user-modified configuration
    config_file = open(config_path, "r")
    config_dict = json.load(config_file)
    config_file.close()

    config = cast(CONFIG, CONFIG.from_dict(config_dict))

    from experiments.env import set_jax_config

    # Set up JAX configuration
    set_jax_config(debug_nans=config.debug_nans)

    import jax
    import torch

    if config.gpu is not None:
        torch.set_default_device(f"cuda:{config.gpu}")
        jax.default_device = jax.devices("gpu")[config.gpu]

    import os
    import wandb

    from jax import numpy as jnp
    # from sbx import PPO
    from stable_baselines3 import PPO

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.rollout import evaluate_policy

    from utils.exp import convert_arg_array

    # Initialize environments and parameters
    Env = get_env(config.env_name)
    sim_env = Env(num_envs=config.baseline_num_envs)
    preal_env = Env(num_envs=config.adapt_num_envs)

    sim_env_conf = sim_env.env
    preal_env_conf = preal_env.env

    # Get default parameter and parameter range
    default_parameter = preal_env_conf.get_parameter()
    num_parameters = default_parameter.shape[0]
    parameter_range = preal_env_conf.parameter_range
    parameter_min, parameter_max = parameter_range

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
    preal_env_conf.model = preal_env_conf.set_parameter(
        preal_env_conf.model, target_parameter
    )

    # Set up evaluation environments
    sim_eval_env = Env(num_envs=config.eval_num_episodes)
    preal_eval_env = Env(num_envs=config.eval_num_episodes)

    sim_eval_env.env.model = sim_env_conf.model
    preal_eval_env.env.model = preal_env_conf.model

    print(f"Target parameter: {target_parameter}")
    print(f"Default parameter: {default_parameter}")
    print()

    # Define evaluation callback function
    def get_eval_callback(eval_env, prefix: str = ""):
        steps = 0
        eval_steps = 0

        prefix = prefix + "_" if len(prefix) > 0 else ""

        def eval(*args, **kwargs):
            nonlocal steps, eval_steps

            steps += config.adapt_num_envs
            if steps - eval_steps < config.eval_num_steps:
                return True
            eval_steps = steps - (steps % config.eval_num_steps)

            model: PPO = args[0]["self"]
            eval_stats = evaluate_policy(eval_env, model, config.eval_num_episodes)

            if config.log_wandb:
                metrics = {
                    "timestep": steps,
                    f"{prefix}eval_mean": eval_stats[0],
                    f"{prefix}eval_std": eval_stats[1],
                }
                wandb.log(metrics)

            return True

        return eval

    # Train baseline model
    model_name = f"PPO-{config.env_name}-baseline"
    model = PPO("MlpPolicy", sim_env, verbose=0)
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if os.path.exists(model_path) and not config.override:
        print(f"Loading baseline model from {model_path}")
        model = model.load(model_path, sim_env)
    else:
        # Initialize wandb logging for baseline training
        if config.log_wandb:
            run_name = "-".join([*exp_levels, name, "baseline"])
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": config.env_name,
                    # "baseline_timesteps": baseline_timesteps,
                    "baseline_round_timesteps": config.baseline_round_timesteps,
                    "baseline_num_envs": config.baseline_num_envs,
                    "param_deviations": config.param_deviations,
                    "adapt_train_lr": config.adapt_train_lr,
                    "adapt_timesteps": config.adapt_timesteps,
                    "adapt_num_envs": config.adapt_num_envs,
                    "eval_num_steps": config.eval_num_steps,
                    "eval_num_episodes": config.eval_num_episodes,
                },
            )

        # Train baseline model until performance threshold is reached
        eval_callback = get_eval_callback(sim_eval_env, prefix="baseline")
        while True:
            model.learn(
                total_timesteps=config.baseline_round_timesteps,
                callback=eval_callback,
                progress_bar=True,
            )
            baseline_eval = evaluate_policy(
                sim_eval_env, model, config.eval_num_episodes
            )
            if baseline_eval[0] >= config.baseline_threshold:
                print(f"Baseline evaluation: {baseline_eval}")
                print()

                break

        model.save(model_path)

    # Start adaptation experiments
    for i in range(config.exp_start, config.exp_start + config.num_exp):
        try:
            # Initialize wandb logging for adaptation experiment
            if config.log_wandb:
                run_name = "-".join([*exp_levels, name, f"{i:02d}"])
                wandb.init(
                    project="differentiable-transfer",
                    name=run_name,
                    config={
                        "env_name": config.env_name,
                        "id": i,
                        "baseline_return": baseline_eval[0],
                        "baseline_num_envs": config.baseline_num_envs,
                        "param_deviations": config.param_deviations,
                        "adapt_train_lr": config.adapt_train_lr,
                        "adapt_timesteps": config.adapt_timesteps,
                        "adapt_num_envs": config.adapt_num_envs,
                        "eval_num_steps": config.eval_num_steps,
                        "eval_num_episodes": config.eval_num_episodes,
                    },
                )

            print(f"Running experiment {i}")
            print()

            # Load baseline model and adapt to new environment
            model = PPO("MlpPolicy", preal_env, verbose=0)
            model = model.load(model_path, preal_env)

            # Evaluate initial performance
            eval_stats = evaluate_policy(preal_eval_env, model, config.eval_num_episodes)
            print(f"Initial evaluation stats: {eval_stats}")

            # Adapt model to new environment
            eval_callback = get_eval_callback(preal_eval_env)
            model.learn(
                total_timesteps=config.adapt_timesteps,
                callback=eval_callback,
                progress_bar=True,
            )

            # Evaluate final performance
            eval_stats = evaluate_policy(preal_eval_env, model, config.eval_num_episodes)
            print(f"Final evaluation stats: {eval_stats}")

            del model

            if config.log_wandb:
                wandb.finish()

        except Exception as e:
            if config.log_wandb:
                wandb.finish(exit_code=1)
                raise e

if __name__ == "__main__":
    app()
