import typer
from typing import Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    exp_start: int = typer.Option(0, help="Start index of the experiment"),
    num_exp: int = typer.Option(3, help="Number of experiments"),
    env_name: str = typer.Option("InvertedPendulum-v5", help="Name of the environment"),
    baseline_timesteps: int = typer.Option(
        5e5, help="Number of timesteps to train baseline models"
    ),
    baseline_num_envs: int = typer.Option(16, help="Number of parallel environments"),
    adapt_param: int = typer.Option(0, help="Index of parameter to adapt"),
    param_deviation: float = typer.Option(
        0.3, help="Deviation from the default parameter"
    ),
    param_value: Optional[float] = typer.Option(
        None, help="Value of the parameter to adapt"
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
    from experiments.env import set_env_vars

    set_env_vars(jax_debug_nans=debug_nans)

    import os
    import wandb

    # from sbx import PPO
    from stable_baselines3 import PPO

    from constants import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    from diff_trans.envs.gym import get_env
    from diff_trans.utils.rollout import evaluate_policy

    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_assets(ROOT_DIR, exp_levels, name)

    # Setup envs and parameters
    Env = get_env(env_name)
    sim_env = Env(num_envs=baseline_num_envs)
    preal_env = Env(num_envs=adapt_num_envs)

    sim_env_conf = sim_env.diff_env
    preal_env_conf = preal_env.diff_env

    default_parameter = preal_env_conf.get_parameter()
    parameter_range = preal_env_conf.parameter_range
    parameter_min, parameter_max = parameter_range

    # Set parameter p to target value
    default_param = default_parameter[adapt_param]
    if param_value is not None:
        target_param = param_value
    else:
        target_param = default_param + param_deviation * (
            parameter_max[adapt_param] - default_param
        )

    target_parameter = default_parameter.at[adapt_param].set(target_param)
    preal_env_conf.model = preal_env_conf.set_parameter(
        preal_env_conf.model, target_parameter
    )

    # env for evaluation
    sim_eval_env = Env(num_envs=eval_num_episodes)
    preal_eval_env = Env(num_envs=eval_num_episodes)
    
    sim_eval_env.diff_env.model = sim_env_conf.model
    preal_eval_env.diff_env.model = preal_env_conf.model

    print(f"Adapting parameter {adapt_param}")
    print(f"Target parameter: {target_param}")
    print(f"Default parameter: {default_param}")
    print()

    # Train baseline model
    model_name = f"PPO-{env_name}-baseline-{adapt_param:02d}"
    model = PPO("MlpPolicy", sim_env, verbose=0)
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if os.path.exists(model_path) and not override:
        print(f"Loading baseline model from {model_path}")
        model = model.load(model_path, sim_env)
    else:
        model.learn(total_timesteps=baseline_timesteps, progress_bar=True)
        model.save(model_path)

    baseline_eval_stats = evaluate_policy(sim_eval_env, model, eval_num_episodes)
    print(f"Baseline evaluation stats: {baseline_eval_stats}")
    print()

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
                    "baseline_return": baseline_eval_stats[0],
                    "baseline_timesteps": baseline_timesteps,
                    "baseline_num_envs": baseline_num_envs,
                    "adapt_param": adapt_param,
                    "param_deviation": param_deviation,
                    "param_value": param_value,
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

        steps = 0
        eval_steps = 0

        def eval(*args, **kwargs):
            nonlocal steps, eval_steps

            # n_steps = args[0]["n_steps"]
            steps += adapt_num_envs
            if steps - eval_steps < eval_num_steps:
                return True
            eval_steps = steps - (steps % eval_num_steps)

            model: PPO = args[0]["self"]
            eval_stats = evaluate_policy(preal_eval_env, model, eval_num_episodes)
            # print(f"Evaluation stats: {steps}, {eval_stats}")

            if log_wandb:
                metrics = dict(
                    timestep=steps,
                    eval_mean=eval_stats[0],
                    eval_std=eval_stats[1],
                )
                # wandb.log(metrics, step=steps, commit=False)
                wandb.log(metrics)

            return True

        model.learn(total_timesteps=adapt_timesteps, callback=eval, progress_bar=True)

        # Evaluate model
        eval_stats = evaluate_policy(preal_eval_env, model, eval_num_episodes)
        print(f"Final evaluation stats: {eval_stats}")

        if log_wandb:
            # wandb.log({}, commit=True)
            wandb.finish()

        del model


if __name__ == "__main__":
    app()
