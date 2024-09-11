import typer
from typing import Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    exp_start: int = typer.Option(0, help="Start index of the experiment"),
    num_exp: int = typer.Option(3, help="Number of experiments"),
    env_name: str = typer.Option("InvertedPendulum-v1", help="Name of the environment"),
    # baseline_timesteps: int = typer.Option(
    #     5e5, help="Number of timesteps to train baseline models"
    # ),
    baseline_round_timesteps: int = typer.Option(
        5e4, help="Number of timesteps to check threshold"
    ),
    baseline_threshold: float = typer.Option(150, help="Threshold of evaluation return"),
    baseline_num_envs: int = typer.Option(16, help="Number of parallel environments"),
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
    override: bool = typer.Option(False, help="Override existing models and logs"),
    debug_nans: bool = typer.Option(False, help="Debug nans"),
):
    from experiments.env import set_jax_config

    set_jax_config(debug_nans=debug_nans)

    import os
    import wandb

    from jax import numpy as jnp

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
    sim_env = Env(num_envs=baseline_num_envs)
    preal_env = Env(num_envs=adapt_num_envs)

    sim_env_conf = sim_env.env
    preal_env_conf = preal_env.env

    default_parameter = preal_env_conf.get_parameter()
    parameter_range = preal_env_conf.parameter_range
    parameter_min, parameter_max = parameter_range

    # Set parameter p to target value
    target_parameter = default_parameter + param_deviation * (
        parameter_max - default_parameter
    )
    preal_env_conf.model = preal_env_conf.set_parameter(
        preal_env_conf.model, target_parameter
    )

    # env for evaluation
    sim_eval_env = Env(num_envs=eval_num_episodes)
    preal_eval_env = Env(num_envs=eval_num_episodes)

    sim_eval_env.env.model = sim_env_conf.model
    preal_eval_env.env.model = preal_env_conf.model

    print(f"Target parameter: {target_parameter}")
    print(f"Default parameter: {default_parameter}")
    print()

    def get_eval_callback(eval_env, prefix: str = ""):
        steps = 0
        eval_steps = 0

        prefix = prefix + "_" if len(prefix) > 0 else ""

        def eval(*args, **kwargs):
            nonlocal steps, eval_steps

            steps += adapt_num_envs
            if steps - eval_steps < eval_num_steps:
                return True
            eval_steps = steps - (steps % eval_num_steps)

            model: PPO = args[0]["self"]
            eval_stats = evaluate_policy(eval_env, model, eval_num_episodes)

            if log_wandb:
                metrics = {
                    "timestep": steps,
                    f"{prefix}eval_mean": eval_stats[0],
                    f"{prefix}eval_std": eval_stats[1],
                }
                wandb.log(metrics)

            return True

        return eval

    # Train baseline model
    model_name = f"PPO-{env_name}-baseline"
    model = PPO("MlpPolicy", sim_env, verbose=0)
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if os.path.exists(model_path) and not override:
        print(f"Loading baseline model from {model_path}")
        model = model.load(model_path, sim_env)
    else:
        if log_wandb:
            run_name = "-".join([*exp_levels, name, "baseline"])
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": env_name,
                    # "baseline_timesteps": baseline_timesteps,
                    "baseline_round_timesteps": baseline_round_timesteps,
                    "baseline_num_envs": baseline_num_envs,
                    "param_deviation": param_deviation,
                    "adapt_train_lr": adapt_train_lr,
                    "adapt_timesteps": adapt_timesteps,
                    "adapt_num_envs": adapt_num_envs,
                    "eval_num_steps": eval_num_steps,
                    "eval_num_episodes": eval_num_episodes,
                },
            )

        eval_callback = get_eval_callback(sim_eval_env, prefix="baseline")
        while True:
            model.learn(
                total_timesteps=baseline_round_timesteps,
                callback=eval_callback,
                progress_bar=True,
            )
            baseline_eval = evaluate_policy(sim_eval_env, model, eval_num_episodes)
            if baseline_eval[0] >= baseline_threshold:
                print(f"Baseline evaluation: {baseline_eval}")
                print()

                break

        model.save(model_path)

    # Start adaptation
    for i in range(exp_start, exp_start + num_exp):
        # Create run
        if log_wandb:
            run_name = "-".join([*exp_levels, name, f"{i:02d}"])
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": env_name,
                    "id": i,
                    "baseline_return": baseline_eval[0],
                    "baseline_num_envs": baseline_num_envs,
                    "param_deviation": param_deviation,
                    "adapt_train_lr": adapt_train_lr,
                    "adapt_timesteps": adapt_timesteps,
                    "adapt_num_envs": adapt_num_envs,
                    "eval_num_steps": eval_num_steps,
                    "eval_num_episodes": eval_num_episodes,
                },
            )

        print(f"Running experiment {i}")
        print()

        model = PPO("MlpPolicy", preal_env, verbose=0)
        model = model.load(model_path, preal_env)

        # Evaluate model
        eval_stats = evaluate_policy(preal_eval_env, model, eval_num_episodes)
        print(f"Initial evaluation stats: {eval_stats}")

        eval_callback = get_eval_callback(preal_eval_env)
        model.learn(
            total_timesteps=adapt_timesteps, callback=eval_callback, progress_bar=True
        )

        # Evaluate model
        eval_stats = evaluate_policy(preal_eval_env, model, eval_num_episodes)
        print(f"Final evaluation stats: {eval_stats}")

        if log_wandb:
            wandb.finish()

        del model


if __name__ == "__main__":
    app()
