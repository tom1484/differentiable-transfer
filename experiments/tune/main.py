import typer
from typing import List, Optional, Union, cast, Dict, Any
from dataclasses import dataclass
from dataclasses_json import dataclass_json

app = typer.Typer(pretty_exceptions_show_locals=False)


# Configuration dataclass for experiment settings
@dataclass_json
@dataclass
class CONFIG:
    cuda_visible_devices: Optional[List[str]] = None
    parallel_instances: Optional[int] = None
    start_exp: int = 0
    num_exp: int = 3

    algorithm: str = "PPO"
    algorithm_config: Optional[Dict[str, Any]] = None
    env_name: str = "InvertedPendulum-v5"

    adapt_params: Union[None, int, List[int]] = None
    param_values: Union[None, float, List[float]] = None
    param_deviations: Union[float, List[float]] = 0.3

    max_tune_epochs: int = 10
    loss_rollout_length: int = 1000
    adapt_learning_rate: float = 1e-2

    adapt_max_timesteps: int = 50000
    adapt_threshold: float = 150
    adapt_num_envs: int = 16

    eval_num_episodes: int = 256
    eval_frequency: int = 10000

    log_wandb: bool = True
    debug_nans: bool = False


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    config_path: Optional[str] = typer.Option(
        None, help="Path to the configuration JSON file"
    ),
):
    import os
    import datetime
    import json
    from experiments.utils.exp import load_config

    config, exp_levels, models_dir = load_config(
        __file__, name, CONFIG, config_path=config_path
    )
    if config is None:
        print("Configuration created")
        return

    # Distribute experiments in tmux
    if config.num_exp > 0 and config.parallel_instances is not None:
        # import uuid
        from utils.path import get_exp_module_name
        from utils.tmux import create_grid_window

        os.makedirs("parallel_tmp", exist_ok=True)

        module_name = get_exp_module_name("experiments", __file__)
        commands = ["python", "-m", module_name, name]

        num_exp_per_parallel = config.num_exp // config.parallel_instances
        num_exps = [num_exp_per_parallel for _ in range(config.parallel_instances)]
        for i in range(
            config.num_exp - num_exp_per_parallel * config.parallel_instances
        ):
            num_exps[i] += 1

        window = create_grid_window(name, config.parallel_instances)
        config.parallel_instances = None
        start_exp = config.start_exp

        for i, num_exp in enumerate(num_exps):
            # config_id = uuid.uuid4().hex
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            tmp_config_path = os.path.join(
                "parallel_tmp", f"{name}-{i}-{timestamp}.json"
            )
            tmp_config_file = open(tmp_config_path, "w")

            config.start_exp = start_exp
            config.num_exp = num_exp
            start_exp = start_exp + num_exp
            json.dump(config.to_dict(), tmp_config_file)

            pane = window.panes[i]
            pane_commands = commands.copy()
            pane_commands.append(f"--config-path={tmp_config_path}")
            pane.send_keys(" ".join(pane_commands))

        return

    from experiments.env import set_env_vars

    set_env_vars(
        jax_debug_nans=config.debug_nans,
        cuda_visible_devices=config.cuda_visible_devices,
    )

    import jax
    import torch

    import os
    import wandb

    import jax.numpy as jnp
    import optax

    # from sbx import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions, evaluate_policy
    from diff_trans.utils.callbacks import (
        StopTrainingOnRewardThreshold,
        EvalCallback,
    )

    from utils.exp import convert_arg_array
    from constants import ALGORITHMS

    Algorithm = ALGORITHMS[config.algorithm]

    exp_start = config.start_exp
    exp_end = exp_start + 1
    if config.num_exp > 0:
        exp_end = exp_start + config.num_exp

    def create_env(Env, parameter: Optional[jnp.ndarray] = None):
        env = Env(num_envs=config.adapt_num_envs)
        env_conf = env.env

        if parameter is not None:
            env_conf.model = env_conf.set_parameter(parameter)

        # env for evaluation
        eval_env = Env(num_envs=config.eval_num_episodes)
        eval_env.env.model = env_conf.model

        return env, env_conf, eval_env

    # Get default parameter and parameter range
    Env = get_env(config.env_name)
    sim_env, sim_env_conf, sim_eval_env = create_env(Env)

    default_parameter = sim_env_conf.get_parameter()
    num_parameters = default_parameter.shape[0]
    parameter_range = sim_env_conf.parameter_range
    parameter_min, parameter_max = parameter_range

    # Determine parameters to adapt and their values
    if config.adapt_params is None:
        adapt_param_ids = jnp.arange(0, num_parameters)
    else:
        adapt_param_ids = convert_arg_array(config.adapt_params, int)

    num_adapt_parameters = adapt_param_ids.shape[0]
    if config.param_values is not None:
        values = convert_arg_array(config.param_values, float, num_adapt_parameters)
    else:
        deviations = convert_arg_array(
            config.param_deviations, float, num_adapt_parameters
        )
        values = (
            default_parameter[adapt_param_ids]
            + deviations * (parameter_max - default_parameter)[adapt_param_ids]
        )

    target_parameter = default_parameter.at[adapt_param_ids].set(values)

    for exp_id in range(exp_start, exp_end):
        # Create run
        if config.log_wandb:
            run_name = "-".join([*exp_levels, name, f"{exp_id:02d}"])
            tags = [
                *exp_levels,
                name,
                f"{exp_id:02d}",
                config.env_name,
                config.algorithm,
            ]
            run = wandb.init(
                project="differentiable-transfer",
                name=run_name,
                tags=tags,
                config={
                    "id": exp_id,
                    "algotithm": config.algorithm,
                    "env_name": config.env_name,
                    "param_deviations": config.param_deviations,
                    "max_tune_epochs": config.max_tune_epochs,
                    "loss_rollout_length": config.loss_rollout_length,
                    "adapt_learning_rate": config.adapt_learning_rate,
                    "adapt_max_timesteps": config.adapt_max_timesteps,
                    "adapt_threshold": config.adapt_threshold,
                    "adapt_num_envs": config.adapt_num_envs,
                    "eval_num_episodes": config.eval_num_episodes,
                },
            )

        try:
            # Setup envs and parameters
            preal_env, _, preal_eval_env = create_env(Env, parameter=target_parameter)

            print(f"Target parameter: {target_parameter}")
            print(f"Default parameter: {default_parameter}")
            print()

            def loss_of_single_param(parameter, rollouts):
                return single_transition_loss(sim_env_conf, parameter, rollouts)

            compute_loss_g = jax.grad(loss_of_single_param, argnums=0)

            # Start parameter tuning
            print(f"Experiment {exp_id}")
            print()

            parameter = default_parameter.copy()
            parameter_subset = parameter[adapt_param_ids]
            optimizer = optax.adam(config.adapt_learning_rate)
            opt_state = optimizer.init(parameter_subset)
            preal_steps = 0

            for i in range(config.max_tune_epochs):
                print(f"Iteration {i}")
                print()

                model_name = (
                    f"{config.algorithm}-{config.env_name}-{exp_id:02d}-{i:02d}"
                )
                model = Algorithm(
                    "MlpPolicy", sim_env, verbose=0, **config.algorithm_config
                )
                model_path = os.path.join(models_dir, f"{model_name}.zip")

                callback_on_best = StopTrainingOnRewardThreshold(
                    reward_threshold=config.adapt_threshold, verbose=1
                )
                eval_callback = EvalCallback(
                    sim_eval_env,
                    n_eval_episodes=config.eval_num_episodes,
                    callback_on_new_best=callback_on_best,
                    # callback_on_log=callback_on_log if config.log_wandb else None,
                    eval_freq=config.eval_frequency // sim_env.num_envs,
                    verbose=0,
                )

                model.learn(
                    total_timesteps=config.adapt_max_timesteps,
                    callback=eval_callback,
                    progress_bar=True,
                )
                # Evaluate model
                sim_eval = evaluate_policy(
                    sim_eval_env, model, config.eval_num_episodes
                )
                print(f"Sim eval: {sim_eval}")
                preal_eval = evaluate_policy(
                    preal_eval_env, model, config.eval_num_episodes
                )
                print(f"Preal eval: {preal_eval}\n")
                model.save(model_path)

                rollouts = rollout_transitions(
                    preal_env, model, num_transitions=config.loss_rollout_length
                )
                preal_steps += config.loss_rollout_length

                loss = loss_of_single_param(parameter, rollouts)
                print(f"Loss: {loss}")
                param_err = (parameter - target_parameter).mean()
                print(f"Parameter error: {param_err}")

                if config.log_wandb:
                    metrics = dict(
                        iteration=i,
                        preal_steps=preal_steps,
                        sim_eval_mean=sim_eval[0],
                        sim_eval_std=sim_eval[1],
                        preal_eval_mean=preal_eval[0],
                        preal_eval_std=preal_eval[1],
                        loss=loss,
                        param_err=param_err,
                    )
                    wandb.log(metrics)

                grad = compute_loss_g(parameter, rollouts)
                print(f"Grad: {grad}")
                if jnp.isnan(grad).any():
                    print("Nan in grad\n")
                    grad = jnp.zeros_like(grad)

                grad_subset = grad[adapt_param_ids]
                updates, opt_state = optimizer.update(grad_subset, opt_state)
                parameter_subset = optax.apply_updates(parameter_subset, updates)
                parameter = parameter.at[adapt_param_ids].set(parameter_subset)

                print(f"Sim parameter: {parameter}")
                print(f"Preal parameter: {target_parameter}")
                print()

                sim_env_conf.model = sim_env_conf.set_parameter(parameter)
                sim_eval_env.env.model = sim_env_conf.model

                del model

            if config.log_wandb:
                wandb.finish()

        except Exception as e:
            if config.log_wandb:
                wandb.finish(exit_code=1)
            raise e


if __name__ == "__main__":
    app()
