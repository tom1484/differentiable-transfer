import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    env_name: str = typer.Option("InvertedPendulum-v5", help="Name of the environment"),
    start_param: int = typer.Option(0, help="Start index of the parameter"),
    max_tune_epochs: int = typer.Option(5, help="Number of epochs to tune"),
    min_loss: float = typer.Option(1e-4, help="Minimum loss to stop tuning"),
    rollout_length: int = typer.Option(1000, help="Number of transitions to rollout"),
    deviation_ratio: float = typer.Option(
        0.3, help="Deviation from the default parameter"
    ),
    learning_rate: float = typer.Option(
        1e-2, help="Learning rate for parameter tuning"
    ),
    train_num_envs: int = typer.Option(4, help="Number of parallel environments"),
    train_steps: int = typer.Option(5e4, help="Number of sub-steps"),
    evaluate: bool = typer.Option(False, help="Evaluate the model"),
    override: bool = typer.Option(False, help="Override existing models and logs"),
    debug_nans: bool = typer.Option(False, help="Debug nans"),
):
    from experiments.env import set_env_vars

    set_env_vars(jax_debug_nans=debug_nans)

    import os

    import wandb

    import jax
    import jax.numpy as jnp
    import optax

    # from sbx import PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    from constants import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions

    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_assets(ROOT_DIR, exp_levels, name)

    # Setup envs and parameters
    env_type = get_env(env_name)
    preal_env = env_type(num_envs=train_num_envs)
    env = env_type(num_envs=train_num_envs)

    preal_env_conf = preal_env.env
    env_conf = env.env

    default_parameter = preal_env_conf.get_parameter()
    parameter_range = preal_env_conf.parameter_range

    model_name = "ppo_inverted_pendulum"

    # Start parameter tuning
    for p in range(start_param, default_parameter.shape[0]):
        # Create run
        run_name = "-".join([*exp_levels, name, f"{p:02d}"])
        wandb.init(
            project="differentiable-transfer",
            name=run_name,
            config={
                "env_name": env_name,
                "max_tune_epochs": max_tune_epochs,
                "min_loss": min_loss,
                "rollout_length": rollout_length,
                "deviation_ratio": deviation_ratio,
                "learning_rate": learning_rate,
                "train_num_envs": train_num_envs,
                "train_steps": train_steps,
                "evaluate": evaluate,
            },
        )

        # Set parameter p to target value
        default_param = default_parameter[p]
        target_parameter = default_parameter + deviation_ratio * (
            parameter_range[1] - default_parameter
        )
        target_param = target_parameter[p]

        preal_env_conf.model = preal_env_conf.set_parameter(
            preal_env_conf.model, target_parameter
        )
        env_conf.model = env_conf.set_parameter(env_conf.model, default_parameter)

        param = default_param.copy()
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(param)

        print(f"Tuning parameter {p}")
        print(f"Target parameter: {target_param}")
        print(f"Default parameter: {default_param}")
        print()

        def loss_of_single_param(param, rollouts):
            return single_transition_loss(
                env_conf, default_parameter.at[p].set(param), rollouts
            )

        compute_loss_g = jax.grad(loss_of_single_param, argnums=0)

        for i in range(max_tune_epochs):
            print(f"Iteration {i}")

            model = PPO("MlpPolicy", env, verbose=0)
            model_path = os.path.join(models_dir, f"{model_name}-{p:02d}-{i:02d}.zip")
            if not override and os.path.exists(model_path):
                model = model.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                model.learn(total_timesteps=train_steps, progress_bar=True)
                model.save(model_path)

            if evaluate:
                sim_eval = evaluate_policy(model, env, n_eval_episodes=10, warn=False)
                print(f"Sim eval: {sim_eval}")
                preal_eval = evaluate_policy(
                    model, preal_env, n_eval_episodes=10, warn=False
                )
                print(f"Preal eval: {preal_eval}")

            rollouts = rollout_transitions(
                preal_env, model, num_transitions=rollout_length
            )
            loss = loss_of_single_param(param, rollouts)
            print(f"Loss: {loss}")

            metrics = dict(
                p=p,
                iteration=i,
                loss=loss,
                param_err=(param - target_param) / target_param,
            )
            if evaluate:
                eval_metrics = dict(
                    sim_eval=sim_eval[0],
                    preal_eval=preal_eval[0],
                )
                metrics.update(eval_metrics)
            
            wandb.log(metrics)

            if loss < min_loss:
                break

            grad = compute_loss_g(param, rollouts)
            print(f"Grad: {grad}")
            if jnp.isnan(grad).any():
                print("Nan in grad\n")
                break

            updates, opt_state = optimizer.update(grad, opt_state)
            param = optax.apply_updates(param, updates)
            print(f"Parameter diff: {param - target_param}")
            print()

            env_conf.model = env_conf.set_parameter(
                env_conf.model, default_parameter.at[p].set(param)
            )

            del model
            
        
        wandb.finish()


if __name__ == "__main__":
    app()
