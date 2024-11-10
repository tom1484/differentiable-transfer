from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[1])

import sys
import wandb
from tqdm import tqdm

import numpy as np
from stable_baselines3.sac import SAC

from diff_trans.envs.gym_wrapper import get_env
from diff_trans.utils.rollout import evaluate_policy

EVAL_NUM_EPISODES = 256
CKPT_TIMESTEPS = 34000
MODEL_PATH = f"./models/env_performance/reacher/SAC-{CKPT_TIMESTEPS:06d}"
LOG = True
TARGET = sys.argv[1]
LOG_NAME = f"reacher_{TARGET}"

print(f"Model Path: {MODEL_PATH}")
print(f"Target: {TARGET}")

if LOG:
    wandb.init(
        project="diff_trans-env_performance",
        name=LOG_NAME,
        config=dict(
            model_path=MODEL_PATH,
            eval_num_episodes=EVAL_NUM_EPISODES,
        ),
    )

Env = get_env("Reacher-v5")
env = Env(num_envs=EVAL_NUM_EPISODES)
default_parameters = env.get_model_parameter()

model = SAC.load(MODEL_PATH)

# N = 30
# armature_range = [0.05, 4.0]
# r = (armature_range[1] / armature_range[0]) ** (1 / (N - 1))
# armatures = armature_range[0] * (r ** np.arange(N))

# for armature in armatures:
#     parameters = default_parameters.copy()
#     parameters = parameters.at[0].set(armature)
#     parameters = parameters.at[1].set(armature)
#     env.set_model_parameter(parameters)

#     result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
#     print(f"Armature: {armature:4.6f}, Mean Return: {result[0]:4.2f}")

#     if LOG:
#         wandb.log(
#             dict(
#                 armature=armature,
#                 mean_return=result[0],
#                 std_return=result[1],
#             )
#         )

# arm_mass_range = [0.01, 0.1]
# arm_masses = np.linspace(*arm_mass_range, num=20)

# for arm_mass in arm_masses:
#     parameters = default_parameters.copy()
#     parameters = parameters.at[4].set(arm_mass)
#     parameters = parameters.at[5].set(arm_mass)
#     env.set_model_parameter(parameters)

#     result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
#     print(f"Arm Mass: {arm_mass:4.2f}, Mean Return: {result[0]:4.2f}")

# tip_mass_range = [0.001, 0.01]
# tip_masses = np.linspace(*tip_mass_range, num=20)

# for tip_mass in tip_masses:
#     parameters = default_parameters.copy()
#     parameters = parameters.at[6].set(tip_mass)
#     env.set_model_parameter(parameters)

#     result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
#     print(f"Tip Mass: {tip_mass:4.2f}, Mean Return: {result[0]:4.2f}")

if TARGET == "damping":
    N = 30
    damping_range = [0.05, 2.0]
    r = (damping_range[1] / damping_range[0]) ** (1 / (N - 1))
    dampings = damping_range[0] * (r ** np.arange(N))

    pbar = tqdm(dampings)
    for damping in pbar:
        parameters = default_parameters.copy()
        parameters = parameters.at[2].set(damping)
        parameters = parameters.at[3].set(damping)
        env.set_model_parameter(parameters)

        result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
        pbar.write(f"Damping: {damping:4.6f}, Mean Return: {result[0]:4.2f}")

        if LOG:
            wandb.log(
                dict(
                    damping=damping,
                    mean_return=result[0],
                    std_return=result[1],
                )
            )

if TARGET == "friction_loss":
    N = 30
    friction_loss_range = [0.01, 1.0]
    r = (friction_loss_range[1] / friction_loss_range[0]) ** (1 / (N - 1))
    friction_losses = friction_loss_range[0] * (r ** np.arange(N))

    pbar = tqdm(friction_losses)
    for friction_loss in pbar:
        parameters = default_parameters.copy()
        parameters = parameters.at[4].set(friction_loss)
        parameters = parameters.at[5].set(friction_loss)

        result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
        pbar.write(f"Friction Loss: {friction_loss:4.6f}, Mean Return: {result[0]:4.2f}")

        if LOG:
            wandb.log(
                dict(
                    friction_loss=friction_loss,
                    mean_return=result[0],
                    std_return=result[1],
                )
            )
