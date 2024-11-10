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
CKPT_TIMESTEP = 1000000
MODEL_PATH = f"./models/env_performance/half_cheetah/SAC-{CKPT_TIMESTEP:07d}"
LOG = True
TARGET = sys.argv[1]
LOG_NAME = f"half_cheetah-{CKPT_TIMESTEP}-{TARGET}"

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

Env = get_env("HalfCheetah-v5")
env = Env(
    num_envs=EVAL_NUM_EPISODES,
    precompile=False,
    exclude_current_positions_from_observation=False,
)
default_parameters = env.get_model_parameter()

model = SAC.load(MODEL_PATH, env=env)

if TARGET == "armature":
    N = 15
    default_armature = default_parameters[1]
    armature_range = [0.05, 0.6]
    r = (armature_range[1] / armature_range[0]) ** (1 / (N - 1))
    armatures = armature_range[0] * (r ** np.arange(N))
    armatures = np.append(armatures, default_armature)

    pbar = tqdm(armatures)
    for armature in pbar:
        parameters = default_parameters.copy()
        parameters = parameters.at[1].set(armature)
        parameters = parameters.at[2].set(armature)
        env.set_model_parameter(parameters)

        result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
        pbar.write(f"Armature: {armature:4.6f}, Mean Return: {result[0]:4.2f}")

        if LOG:
            wandb.log(
                dict(
                    armature=armature,
                    mean_return=result[0],
                    std_return=result[1],
                )
            )

if TARGET == "torso_mass":
    N = 15
    default_torso_mass = default_parameters[5]
    torso_mass_range = [1, 15]
    # r = (torso_mass_range[1] / torso_mass_range[0]) ** (1 / (N - 1))
    # torso_masses = torso_mass_range[0] * (r ** np.arange(N))
    torso_masses = np.linspace(torso_mass_range[0], torso_mass_range[1], N)
    torso_masses = np.append(torso_masses, default_torso_mass)

    pbar = tqdm(torso_masses)
    for torso_mass in pbar:
        parameters = default_parameters.copy()
        parameters = parameters.at[5].set(torso_mass)
        env.set_model_parameter(parameters)

        result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
        pbar.write(f"Torso Mass: {torso_mass:4.6f}, Mean Return: {result[0]:4.2f}")

        if LOG:
            wandb.log(
                dict(
                    torso_mass=torso_mass,
                    mean_return=result[0],
                    std_return=result[1],
                )
            )

# if TARGET == "damping":
#     N = 15
#     default_dampings = default_parameters[3:5]
#     damping_ranges = [[1.0, 20.0], [0.75, 15.0]]

#     dampings = []
#     for i, damping_range in enumerate(damping_ranges):
#         r = (damping_range[1] / damping_range[0]) ** (1 / (N - 1))
#         dampings.append(
#             np.append(
#                 damping_range[0] * (r ** np.arange(N)),
#                 default_dampings[i],
#             )
#         )

#     dampings = np.array(dampings).T

#     pbar = tqdm(dampings)
#     for damping in pbar:
#         parameters = default_parameters.copy()
#         parameters = parameters.at[3].set(damping[0])
#         parameters = parameters.at[4].set(damping[1])

#         result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
#         pbar.write(
#             f"Damping: [{damping[0]:4.6f}, {damping[1]:4.6f}], Mean Return: {result[0]:4.2f}"
#         )

#         if LOG:
#             wandb.log(
#                 dict(
#                     damping0=damping[0],
#                     damping1=damping[1],
#                     mean_return=result[0],
#                     std_return=result[1],
#                 )
#             )

if TARGET == "damping":
    N = 15
    default_dampings = default_parameters[3:5]
    ratio_range = [0.5, 2.0]
    ratios = np.linspace(ratio_range[0], ratio_range[1], N)
    ratios = np.append(ratios, 1.0)

    pbar = tqdm(ratios)
    for ratio in pbar:
        parameters = default_parameters.copy()
        parameters = parameters.at[3].set(default_dampings[0] * ratio)
        parameters = parameters.at[4].set(default_dampings[1] * ratio)

        result = evaluate_policy(env, model, n_eval_episodes=EVAL_NUM_EPISODES)
        pbar.write(f"Damping Ratio: {ratio:4.6f}, Mean Return: {result[0]:4.2f}")

        if LOG:
            wandb.log(
                dict(
                    damping_ratio=ratio,
                    mean_return=result[0],
                    std_return=result[1],
                )
            )
