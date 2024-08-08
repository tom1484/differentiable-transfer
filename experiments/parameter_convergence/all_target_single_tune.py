from experiments.env import set_jax_config


set_jax_config()


import typer
import os
import traceback

# import wandb

import jax
import jax.numpy as jnp
import optax

# from sbx import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from definitions import ROOT_DIR
from utils.path import get_exp_file_levels, create_exp_dirs

from diff_trans.envs.wrapped import get_env
from diff_trans.utils.loss import single_transition_loss
from diff_trans.utils.rollout import rollout_transitions


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    env_name: str = typer.Option("InvertedPendulum-v1", help="Name of the environment"),
    tune_epochs: int = typer.Option(5, help="Number of epochs to train"),
    rollout_length: int = typer.Option(1000, help="Number of transitions to rollout"),
    deviation_ratio: float = typer.Option(
        0.3, help="Deviation from the default parameter"
    ),
    learning_rate: float = typer.Option(
        1e-1, help="Learning rate for parameter tuning"
    ),
    train_num_envs: int = typer.Option(4, help="Number of parallel environments"),
    train_steps: int = typer.Option(5e4, help="Number of sub-steps"),
    evaluate: bool = typer.Option(False, help="Evaluate the model"),
    override: bool = typer.Option(False, help="Override existing models and logs"),
):
    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_dirs(ROOT_DIR, exp_levels, name)

    # Setup envs and parameters
    env_type = get_env(env_name)
    preal_env = env_type(num_envs=train_num_envs)
    env = env_type(num_envs=train_num_envs)

    preal_env_conf = preal_env.env
    env_conf = env.env

    default_parameter = preal_env_conf.get_parameter()
    parameter_range = preal_env_conf.parameter_range

    model_name = "ppo_inverted_pendulum"

    compute_loss_vg = jax.value_and_grad(single_transition_loss, argnums=1)

    # Start parameter tuning
    # Set start index to 0 in the future
    for p in range(2, default_parameter.shape[0]):
        # Set parameter p to target value
        default_param = default_parameter[p]
        target_param = default_param + deviation_ratio * (
            parameter_range[1][p] - default_param
        )
        target_parameter = default_parameter.at[p].set(target_param)
        preal_env_conf.model = preal_env_conf.set_parameter(
            preal_env_conf.model, target_parameter
        )

        parameter = default_parameter.copy()
        env_conf.model = env_conf.set_parameter(env_conf.model, parameter)

        # optimizer = optax.adam(learning_rate)
        # opt_state = optimizer.init(parameter)

        print(f"Tuning parameter {p}")
        print(f"Target parameter: {target_parameter}")
        print(f"Default parameter: {default_parameter}")
        print()

        # for i in range(tune_epochs):
        #     print(f"Iteration {i}")

        model = PPO("MlpPolicy", env, verbose=0)
        model_path = os.path.join(models_dir, f"{model_name}-{p:02d}-00.zip")
        if not override and os.path.exists(model_path):
            model = model.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            model.learn(total_timesteps=train_steps, progress_bar=True)
            model.save(model_path)

        rollouts = rollout_transitions(preal_env, model, num_transitions=rollout_length)
        loss, grad = compute_loss_vg(env_conf, parameter, rollouts)

        # updates, opt_state = optimizer.update(grad, opt_state)
        # param = optax.apply_updates(param, updates)
        # env_conf.model = env_conf.set_parameter(
        #     env_conf.model, default_parameter.at[p].set(param)
        # )

        del model

        print(f"Loss: {loss}")
        print(f"Grad: ", end="")
        print(grad)
        # print(f"Parameter diff: ", end="")
        # print(param - target_param)

        print()


if __name__ == "__main__":
    app()
