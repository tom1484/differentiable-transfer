import typer
from typing import Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    parallel: bool = typer.Argument(False, help="Create parallel tmux sessions"),
    exp_id: int = typer.Option(0, help="Experiment ID"),
    num_exp: int = typer.Option(3, help="Number of experiments"),
    start_exp_id: int = typer.Option(0, help="Start experiment ID"),
    env_name: str = typer.Option("InvertedPendulum-v1", help="Name of the environment"),
    adapt_param: int = typer.Option(0, help="Index of parameter to adapt"),
    param_deviation: float = typer.Option(
        0.3, help="Deviation from the default parameter"
    ),
    param_value: Optional[float] = typer.Option(
        None, help="Value of the parameter to adapt"
    ),
    max_tune_epochs: int = typer.Option(10, help="Number of epochs to tune"),
    loss_rollout_length: int = typer.Option(
        1000, help="Number of transitions to rollout"
    ),
    adapt_learning_rate: float = typer.Option(
        1e-2, help="Learning rate for parameter tuning"
    ),
    # adapt_timesteps: int = typer.Option(5e5, help="Number of timesteps to adapt"),
    adapt_round_timesteps: int = typer.Option(
        5e4, help="Number of timesteps to check threshold"
    ),
    adapt_threshold: float = typer.Option(150, help="Threshold of evaluation return"),
    adapt_num_envs: int = typer.Option(16, help="Number of parallel environments"),
    log_wandb: bool = typer.Option(True, help="Log to wandb"),
    eval_num_steps: int = typer.Option(1e4, help="Number of steps to evaluate"),
    eval_num_episodes: int = typer.Option(256, help="Number of episodes to evaluate"),
    debug_nans: bool = typer.Option(False, help="Debug nans"),
):
    # Distribute experiments in tmux
    if num_exp > 0 and parallel:
        from utils.cmd import args_to_commands
        from utils.path import get_exp_module_name
        from utils.tmux import create_grid_window

        module_name = get_exp_module_name("experiments", __file__)
        commands = ["python", "-m", module_name, name]
        commands.extend(
            args_to_commands(
                env_name=env_name,
                param_deviation=param_deviation,
                max_tune_epochs=max_tune_epochs,
                loss_rollout_length=loss_rollout_length,
                adapt_learning_rate=adapt_learning_rate,
                adapt_threshold=adapt_threshold,
                adapt_round_timesteps=adapt_round_timesteps,
                adapt_num_envs=adapt_num_envs,
                eval_num_steps=eval_num_steps,
                eval_num_episodes=eval_num_episodes,
                debug_nans=debug_nans,
            )
        )

        window = create_grid_window(name, num_exp)
        for exp_id in range(start_exp_id, start_exp_id + num_exp):
            pane = window.panes[exp_id]
            pane_commands = commands.copy()
            pane_commands.append(f"--exp-id={exp_id}")
            pane.send_keys(" ".join(pane_commands))

        return

    from experiments.env import set_jax_config

    set_jax_config(debug_nans=debug_nans)

    import os
    import wandb

    import jax
    import jax.numpy as jnp
    import optax

    # from sbx import PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    from definitions import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions, evaluate_policy

    # Create folders for the experiment
    exp_levels = get_exp_file_levels("experiments", __file__)
    logs_dir, models_dir = create_exp_assets(ROOT_DIR, exp_levels, name)

    exp_start = exp_id
    exp_end = exp_id + 1
    if num_exp > 0:
        exp_end = exp_id + num_exp

    def create_env(Env, parameter: Optional[jnp.ndarray] = None):
        env = Env(num_envs=adapt_num_envs)
        env_conf = env.env

        if parameter is not None:
            env_conf.model = env_conf.set_parameter(env_conf.model, parameter)

        # env for evaluation
        eval_env = Env(num_envs=eval_num_episodes)
        eval_env.env.model = env_conf.model

        return env, env_conf, eval_env

    for exp_id in range(exp_start, exp_end):
        # Setup envs and parameters
        Env = get_env(env_name)
        sim_env, sim_env_conf, sim_eval_env = create_env(Env)

        default_parameter = sim_env_conf.get_parameter()
        parameter_range = sim_env_conf.parameter_range
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
        preal_env, _, preal_eval_env = create_env(Env, parameter=target_parameter)

        print(f"Adapting parameter {adapt_param}")
        print(f"Target parameter: {target_param}")
        print(f"Default parameter: {default_param}")
        print()

        def loss_of_single_param(param, rollouts):
            return single_transition_loss(
                sim_env_conf, default_parameter.at[adapt_param].set(param), rollouts
            )
        
        compute_loss_g = jax.grad(loss_of_single_param, argnums=0)

        print(f"Experiment {exp_id}")
        print()

        # Create run
        if log_wandb:
            run_name = "-".join([*exp_levels, name, f"{exp_id:02d}"])
            wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": env_name,
                    "id": exp_id,
                    "adapt_param": adapt_param,
                    "param_deviation": param_deviation,
                    "param_value": param_value,
                    "max_tune_epochs": max_tune_epochs,
                    "loss_rollout_length": loss_rollout_length,
                    "adapt_learning_rate": adapt_learning_rate,
                    # "adapt_timesteps": adapt_timesteps,
                    "adapt_round_timesteps": adapt_round_timesteps,
                    "adapt_num_envs": adapt_num_envs,
                    "eval_num_steps": eval_num_steps,
                    "eval_num_episodes": eval_num_episodes,
                },
            )
        
        param = default_param.copy()
        optimizer = optax.adam(adapt_learning_rate)
        opt_state = optimizer.init(param)
        preal_steps = 0

        for i in range(max_tune_epochs):
            print(f"Iteration {i}")
            print()

            model = PPO("MlpPolicy", sim_env, verbose=0)
            model_name = f"PPO-{env_name}-{adapt_param:02d}-{exp_id:02d}-{i:02d}"
            model_path = os.path.join(models_dir, f"{model_name}.zip")
            
            if os.path.exists(model_path):
                model = model.load(model_path)
            else:
                while True:
                    model.learn(
                        total_timesteps=adapt_round_timesteps, progress_bar=True
                    )
                    sim_eval = evaluate_policy(sim_eval_env, model, eval_num_episodes)
                    if sim_eval[0] >= adapt_threshold:
                        # Evaluate model
                        # sim_eval = evaluate_policy(sim_eval_env, model, eval_num_episodes)
                        print(f"Sim eval: {sim_eval}")
                        preal_eval = evaluate_policy(
                            preal_eval_env, model, eval_num_episodes
                        )
                        print(f"Preal eval: {preal_eval}")
                        print()

                        break

                model.save(model_path)

            rollouts = rollout_transitions(
                preal_env, model, num_transitions=loss_rollout_length
            )
            preal_steps += loss_rollout_length

            loss = loss_of_single_param(param, rollouts)
            print(f"Loss: {loss}")

            if log_wandb:
                metrics = dict(
                    iteration=i,
                    preal_steps=preal_steps,
                    sim_eval_mean=sim_eval[0],
                    sim_eval_std=sim_eval[1],
                    preal_eval_mean=preal_eval[0],
                    preal_eval_std=preal_eval[1],
                    loss=loss,
                    param_err=(param - target_param) / target_param,
                )
                wandb.log(metrics)

            grad = compute_loss_g(param, rollouts)
            print(f"Grad: {grad}")
            if jnp.isnan(grad).any():
                print("Nan in grad\n")
                break

            updates, opt_state = optimizer.update(grad, opt_state)
            param = optax.apply_updates(param, updates)
            print(f"Parameter diff: {param - target_param}")
            print()

            sim_env_conf.model = sim_env_conf.set_parameter(
                sim_env_conf.model, default_parameter.at[adapt_param].set(param)
            )
            sim_eval_env.env.model = sim_env_conf.model

            del model

        if log_wandb:
            wandb.finish()


if __name__ == "__main__":
    app()
