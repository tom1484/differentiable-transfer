import typer
from typing import List, Optional, Union, cast
from dataclasses import dataclass
from dataclasses_json import dataclass_json

app = typer.Typer(pretty_exceptions_show_locals=False)


# Configuration dataclass for experiment settings
@dataclass_json
@dataclass
class CONFIG:
    name: str
    gpu: Optional[int] = None
    parallel_instances: Optional[int] = None
    num_exp: int = 3
    start_exp: int = 0
    env_name: str = "InvertedPendulum-v1"
    adapt_params: Union[None, int, List[int]] = None
    param_values: Union[None, float, List[float]] = None
    param_deviations: Union[float, List[float]] = 0.3
    max_tune_epochs: int = 10
    loss_rollout_length: int = 1000
    adapt_learning_rate: float = 1e-2
    adapt_round_timesteps: int = 5e4
    adapt_threshold: float = 150
    adapt_num_envs: int = 16
    log_wandb: bool = True
    eval_num_steps: int = 1e4
    eval_num_episodes: int = 256
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
    from definitions import ROOT_DIR
    from utils.path import get_exp_file_levels, create_exp_assets

    # Initialize default configuration
    default_config = CONFIG(name=name)

    # Create experiment assets (folders and default configuration)
    exp_levels = get_exp_file_levels("experiments", __file__)
    new_config, default_config_path, models_dir = create_exp_assets(
        ROOT_DIR, exp_levels, name, default_config.to_dict()
    )

    if config_path is None:
        config_path = default_config_path

    if new_config:
        print("Configuration created")
        return

    # Load user-modified configuration
    config_file = open(config_path, "r")
    config_dict = json.load(config_file)
    config_file.close()

    config = cast(CONFIG, CONFIG.from_dict(config_dict))

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

    from experiments.env import set_jax_config

    set_jax_config(debug_nans=config.debug_nans)

    import jax
    import torch

    if config.gpu is not None:
        torch.set_default_device(f"cuda:{config.gpu}")
        jax.default_device = jax.devices("gpu")[config.gpu]

    import os
    import wandb

    import jax.numpy as jnp
    import optax

    # from sbx import PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    from diff_trans.envs.wrapped import get_env
    from diff_trans.utils.loss import single_transition_loss
    from diff_trans.utils.rollout import rollout_transitions, evaluate_policy

    from utils.exp import convert_arg_array

    exp_start = config.start_exp
    exp_end = exp_start + 1
    if config.num_exp > 0:
        exp_end = exp_start + config.num_exp

    def create_env(Env, parameter: Optional[jnp.ndarray] = None):
        env = Env(num_envs=config.adapt_num_envs)
        env_conf = env.env

        if parameter is not None:
            env_conf.model = env_conf.set_parameter(env_conf.model, parameter)

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
            run = wandb.init(
                project="differentiable-transfer",
                name=run_name,
                config={
                    "env_name": config.env_name,
                    "id": exp_id,
                    "param_deviations": config.param_deviations,
                    "max_tune_epochs": config.max_tune_epochs,
                    "loss_rollout_length": config.loss_rollout_length,
                    "adapt_learning_rate": config.adapt_learning_rate,
                    # "adapt_timesteps": adapt_timesteps,
                    "adapt_round_timesteps": config.adapt_round_timesteps,
                    "adapt_threshold": config.adapt_threshold,
                    "adapt_num_envs": config.adapt_num_envs,
                    "eval_num_steps": config.eval_num_steps,
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
            optimizer = optax.adam(config.adapt_learning_rate)
            opt_state = optimizer.init(parameter)
            preal_steps = 0

            for i in range(config.max_tune_epochs):
                print(f"Iteration {i}")
                print()

                model = PPO("MlpPolicy", sim_env, verbose=0)
                model_name = f"PPO-{config.env_name}-{exp_id:02d}-{i:02d}"
                model_path = os.path.join(models_dir, f"{model_name}.zip")

                # if os.path.exists(model_path):
                #     model = model.load(model_path)

                while True:
                    model.learn(
                        total_timesteps=config.adapt_round_timesteps,
                        progress_bar=True,
                    )
                    sim_eval = evaluate_policy(
                        sim_eval_env, model, config.eval_num_episodes
                    )
                    if sim_eval[0] >= config.adapt_threshold:
                        # Evaluate model
                        print(f"Sim eval: {sim_eval}")
                        preal_eval = evaluate_policy(
                            preal_eval_env, model, config.eval_num_episodes
                        )
                        print(f"Preal eval: {preal_eval}")
                        print()

                        break

                model.save(model_path)

                rollouts = rollout_transitions(
                    preal_env, model, num_transitions=config.loss_rollout_length
                )
                preal_steps += config.loss_rollout_length

                loss = loss_of_single_param(parameter, rollouts)
                print(f"Loss: {loss}")

                if config.log_wandb:
                    metrics = dict(
                        iteration=i,
                        preal_steps=preal_steps,
                        sim_eval_mean=sim_eval[0],
                        sim_eval_std=sim_eval[1],
                        preal_eval_mean=preal_eval[0],
                        preal_eval_std=preal_eval[1],
                        loss=loss,
                        param_err=(parameter - target_parameter) / target_parameter,
                    )
                    wandb.log(metrics)

                grad = compute_loss_g(parameter, rollouts)
                print(f"Grad: {grad}")
                if jnp.isnan(grad).any():
                    print("Nan in grad\n")
                    break

                updates, opt_state = optimizer.update(grad, opt_state)
                parameter = optax.apply_updates(parameter, updates)
                print(f"Parameter diff: {parameter - target_parameter}")
                print()

                sim_env_conf.model = sim_env_conf.set_parameter(
                    sim_env_conf.model, parameter
                )
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
