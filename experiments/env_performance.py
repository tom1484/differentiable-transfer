import typer
from typing import List, Optional, Dict, Any

from experiments.utils.config import *
from utils import default

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class Config:
    cuda_visible_devices: Optional[List[str]] = None

    algorithm: str = "SAC"
    algorithm_config: Optional[Dict[str, Any]] = None

    env_name: str = "InvertedPendulum-v5"
    env_config: Optional[Dict[str, Any]] = None

    timesteps: int = 500000
    eval_num_episodes: int = 256
    eval_frequency: int = 5000

    log_wandb: bool = False


# Main entry point for the experiment
@app.command()
def main(name: str = typer.Argument(..., help="Name of the experiment")):
    from experiments.utils.exp import load_config
    from experiments.env import set_env_vars

    config, exp_levels, models_dir = load_config(__file__, name, Config)
    if config is None:
        print("Configuration created")
        return

    set_env_vars(
        cuda_visible_devices=config.cuda_visible_devices,
        jax_platforms="cpu",
    )

    import os
    import wandb

    from diff_trans.envs.gym_wrapper import get_env
    from diff_trans.utils.rollout import evaluate_policy
    from diff_trans.utils.callbacks import EvalCallback, SaveModelCallback

    from constants import ALGORITHMS
    from stable_baselines3.common.vec_env import SubprocVecEnv

    Algorithm = ALGORITHMS[config.algorithm]

    # Initialize environments and parameters
    print("Creating environment... ", end="")
    Env = get_env(config.env_name)
    diff_env = Env(precompile=False)

    create_env = lambda: diff_env.create_gym_env(**default(config.env_config, {}))
    env = create_env()
    eval_env = SubprocVecEnv([create_env for _ in range(32)])
    print("Done")

    # Train baseline model
    model_name = f"{config.algorithm}-{config.env_name}"
    model = Algorithm(
        "MlpPolicy", env, verbose=0, **default(config.algorithm_config, {})
    )
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    if config.log_wandb:
        run_name = "-".join([*exp_levels, name])
        tags = [*exp_levels, name, config.env_name, config.algorithm]
        wandb.init(
            project="diff_trans-env_performance",
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

    save_model_callback = SaveModelCallback(models_dir, base_name=model_name)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config.eval_num_episodes,
        callback_on_log=callback_on_log,
        callback_after_eval=save_model_callback,
        eval_freq=config.eval_frequency,
        verbose=1,
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
