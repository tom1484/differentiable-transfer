import typer
import os

import jax
import optax

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from definitions import ROOT_DIR
from utils.path import get_exp_file_levels, create_exp_dirs

from diff_trans.envs.wrapped import get_env
from diff_trans.utils.loss import single_transition_loss
from diff_trans.utils.rollout import rollout_transitions


def tune(
    name: str = typer.Argument(..., help="Name of the experiment"),
    env_name: str = typer.Option("InvertedPendulum-v1", help="Name of the environment"),
    tune_epochs: int = typer.Option(5, help="Number of epochs to train"),
    learning_rate: float = typer.Option(
        1e-1, help="Learning rate for parameter tuning"
    ),
    train_num_envs: int = typer.Option(4, help="Number of parallel environments"),
    train_steps: int = typer.Option(1e5, help="Number of sub-steps"),
    override: bool = typer.Option(False, help="Override existing models and logs"),
):
    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_dirs(ROOT_DIR, exp_levels, name, override)

    # Setup envs and parameters
    env_type = get_env(env_name)

    preal_env = env_type(num_envs=train_num_envs)
    preal_env_conf = preal_env.env

    default_parameter = preal_env_conf.get_parameter()
    parameter_range = preal_env_conf.parameter_range

    ratio = 0.1
    target_parameter = default_parameter + ratio * (parameter_range[1] - default_parameter)
    preal_env_conf.model = preal_env_conf.set_parameter(
        preal_env_conf.model, target_parameter
    )

    env = env_type(num_envs=train_num_envs)
    env_conf = env.env
    parameter = env_conf.get_parameter()

    # Start parameter tuning

    compute_loss_jit = jax.jit(single_transition_loss, static_argnums=0)
    compute_loss_vg = jax.value_and_grad(compute_loss_jit, argnums=1)

    model_name = "ppo_inverted_pendulum"

    optimizer = optax.adam(1e-1)
    opt_state = optimizer.init(parameter)

    for i in range(5):
        print(f"Iteration {i}")

        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save(os.path.join(models_dir, "{model_name}-{i:02d}"))

        sim_eval = evaluate_policy(model, env, n_eval_episodes=10, warn=False)
        print(f"Sim eval: {sim_eval}")

        preal_eval = evaluate_policy(model, preal_env, n_eval_episodes=10, warn=False)
        print(f"Preal eval: {preal_eval}")
        
        rollouts = rollout_transitions(preal_env, model, num_transitions=1000)
        loss, grad = compute_loss_vg(env_conf, parameter, rollouts)

        updates, opt_state = optimizer.update(grad, opt_state)
        parameter = optax.apply_updates(parameter, updates)
        env_conf.model = env_conf.set_parameter(env_conf.model, parameter)

        del model

        print(f"Loss: {loss}")
        print(f"Grad: {grad}")
        print(f"Parameter: {parameter}")

        print()


if __name__ == "__main__":
    typer.run(tune)
