import typer

from experiments.env import set_env_vars

set_env_vars()


def tune(
    name: str = typer.Argument(..., help="Name of the experiment"),
    env_name: str = typer.Option("InvertedPendulum-v5", help="Name of the environment"),
    train_num_envs: int = typer.Option(4, help="Number of parallel environments"),
    train_steps: int = typer.Option(5e4, help="Number of sub-steps"),
    evaluate: bool = typer.Option(False, help="Evaluate the model"),
    override: bool = typer.Option(False, help="Override existing models and logs"),
):
    import os

    # import wandb

    import jax
    import optax

    # from sbx import PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    from constants import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    from diff_trans.envs.gym import get_env
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions

    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_assets(ROOT_DIR, exp_levels, name)

    # Setup envs and parameters
    env_type = get_env(env_name)

    preal_env = env_type(num_envs=train_num_envs)
    env = env_type(num_envs=train_num_envs)

    preal_env_conf = preal_env.diff_env
    env_conf = env.diff_env

    # Start parameter tuning
    compute_loss_jit = jax.jit(single_transition_loss, static_argnums=0)
    compute_loss_vg = jax.value_and_grad(compute_loss_jit, argnums=1)

    model_name = "ppo_inverted_pendulum"
    model_path = os.path.join(models_dir, f"{model_name}.zip")

    model = PPO("MlpPolicy", env, verbose=0)
    if os.path.exists(model_path):
        model = model.load(model_path)
    else:
        model.learn(total_timesteps=train_steps, progress_bar=True)
        model.save(model_path)

    for _ in range(1):
        # rollouts = rollout_transitions(env, model, num_transitions=1000)
        rollouts = rollout_transitions(preal_env, model, num_transitions=100)
        # loss = compute_loss_jit(env_conf, env_conf.get_parameter(), rollouts)
        loss = single_transition_loss(env_conf, env_conf.get_parameter(), rollouts)
        print(f"Loss: {loss}")


if __name__ == "__main__":
    typer.run(tune)
