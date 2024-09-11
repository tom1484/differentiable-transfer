import typer
from typing import Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    exp_start: int = typer.Option(0, help="Start index of the experiment"),
    num_exp: int = typer.Option(3, help="Number of experiments"),
    env_name: str = typer.Option("InvertedPendulum-v1", help="Name of the environment"),
    param_deviation: float = typer.Option(
        0.3, help="Deviation from the default parameter"
    ),
    adapt_train_lr: float = typer.Option(
        1e-2, help="Learning rate for parameter tuning"
    ),
    adapt_timesteps: int = typer.Option(5e5, help="Number of timesteps to adapt"),
    adapt_num_envs: int = typer.Option(16, help="Number of parallel environments"),
    log_wandb: bool = typer.Option(True, help="Log to wandb"),
    eval_num_steps: int = typer.Option(1e4, help="Number of steps to evaluate"),
    eval_num_episodes: int = typer.Option(256, help="Number of episodes to evaluate"),
    debug_nans: bool = typer.Option(False, help="Debug nans"),
):
    from experiments.env import set_jax_config

    set_jax_config(debug_nans=debug_nans)

    import os
    import time
    import wandb

    from jax import numpy as jnp
    from jax import random as jrandom

    # from sbx import PPO
    from stable_baselines3 import PPO

    from definitions import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.rollout import evaluate_policy

    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_assets(ROOT_DIR, exp_levels, name)

    # Setup envs and parameters
    Env = get_env(env_name)

    def sample_env():
        env = Env(num_envs=adapt_num_envs)
        env_conf = env.env

        default_parameter = env_conf.get_parameter()
        parameter_range = env_conf.parameter_range
        parameter_min, parameter_max = parameter_range

        # Set parameter p to target value
        rng = jrandom.PRNGKey(time.time_ns())
        target_parameter = jrandom.uniform(
            rng,
            shape=default_parameter.shape,
            minval=parameter_min,
            maxval=parameter_max,
        )
        env_conf.model = env_conf.set_parameter(env_conf.model, target_parameter)

        # env for evaluation
        eval_env = Env(num_envs=eval_num_episodes)
        eval_env.env.model = env_conf.model

        return env, eval_env
    
    def get_eval_callback():
        steps = 0
        eval_steps = 0

        def eval(*args, **kwargs):
            nonlocal steps, eval_steps

            steps += adapt_num_envs
            if steps - eval_steps < eval_num_steps:
                return True
            eval_steps = steps - (steps % eval_num_steps)

            model: PPO = args[0]["self"]
            eval_stats = evaluate_policy(eval_env, model, eval_num_episodes)

            if log_wandb:
                metrics = dict(
                    timestep=steps,
                    eval_mean=eval_stats[0],
                    eval_std=eval_stats[1],
                )
                wandb.log(metrics)

            return True
        
        return eval

    # Start adaptation
    for exp_id in range(exp_start, exp_start + num_exp):
        # Sample env
        env, eval_env = sample_env()
        print(env.env.get_parameter())
        
        # continue

        # Create run
        if log_wandb:
            run_name = "-".join([*exp_levels, name, f"{exp_id:02d}"])
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": env_name,
                    "id": exp_id,
                    "parameters": env.env.get_parameter(),
                    "param_deviation": param_deviation,
                    "adapt_train_lr": adapt_train_lr,
                    "adapt_timesteps": adapt_timesteps,
                    "adapt_num_envs": adapt_num_envs,
                    "eval_num_steps": eval_num_steps,
                    "eval_num_episodes": eval_num_episodes,
                },
            )

        print(f"Running experiment {exp_id}")
        print()

        model = PPO("MlpPolicy", env, verbose=0)
        model_name = f"PPO-{env_name}-{exp_id}"
        # model_path = os.path.join(models_dir, f"{model_name}.zip")

        eval_callback = get_eval_callback()
        model.learn(total_timesteps=adapt_timesteps, callback=eval_callback, progress_bar=True)

        if log_wandb:
            wandb.finish()

        del model


if __name__ == "__main__":
    app()
