import typer
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
from dataclasses_json import dataclass_json

app = typer.Typer(pretty_exceptions_show_locals=False)


# Configuration dataclass for experiment settings
@dataclass_json
@dataclass
class CONFIG:
    cuda_visible_devices: Optional[List[str]] = None

    algorithm: str = "PPO"
    algorithm_config: Optional[Dict[str, Any]] = None

    env_name: str = "InvertedPendulum-v5"
    num_envs: int = 256

    timesteps: int = 1000000
    eval_num_episodes: int = 256
    eval_frequency: int = 10000

    log_wandb: bool = False
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

    from diff_trans.envs.gym_wrapper import get_env
    from diff_trans.utils.rollout import evaluate_policy
    from diff_trans.utils.callbacks import EvalCallback

    from constants import ALGORITHMS

    Algorithm = ALGORITHMS[config.algorithm]

    # Initialize environments and parameters
    Env = get_env(config.env_name)
    env = Env(num_envs=config.num_envs)
    env_conf = env.diff_env
    eval_env = Env(num_envs=config.eval_num_episodes)

    # Train baseline model
    model_name = f"{config.algorithm}-{config.env_name}"
    model = Algorithm("MlpPolicy", env, verbose=0, **config.algorithm_config)
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if config.log_wandb:
        run_name = "-".join([*exp_levels, name])
        tags = [*exp_levels, name, config.env_name, config.algorithm]
        wandb.init(
            project="differentiable-transfer",
            name=run_name,
            tags=tags,
            config={
                "env_name": config.env_name,
                "timesteps": config.timesteps,
                "eval_num_episodes": config.eval_num_episodes,
                "eval_frequency": config.eval_frequency,
            },
        )

    def callback_on_log(metrics):
        if config.log_wandb:
            wandb.log(metrics)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config.eval_num_episodes,
        callback_on_log=callback_on_log,
        eval_freq=config.eval_frequency // env.num_envs,
        verbose=0,
    )

    # Train baseline model until performance threshold is reached
    model.learn(
        total_timesteps=config.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    eval_result = evaluate_policy(eval_env, model, config.eval_num_episodes)
    print(f"Evaluation: {eval_result}\n")

    model.save(model_path)
    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    app()
