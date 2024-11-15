from typing import List, Tuple

from tqdm import tqdm

from jax import numpy as jnp
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv

from ..envs.gym_wrapper import BaseEnv


Transition = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
Trajectory = List[Transition]


def squeeze_array_envs(array: jnp.ndarray):
    num_dims = len(array.shape)
    return jnp.transpose(array, (1, 0, *list(range(2, num_dims))))


def rollout_transitions(
    env: BaseEnv | SubprocVecEnv, model, num_transitions=100
) -> List[Trajectory]:
    num_envs = env.num_envs

    num_steps = num_transitions // num_envs
    if num_transitions % num_envs > 0:
        num_steps += 1
    num_transitions = num_steps * num_envs

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []

    observation = env.reset()
    for _ in range(num_steps):
        action, _ = model.predict(observation)
        next_observation, reward, done, _ = env.step(action)

        observations.append(observation)
        next_observations.append(next_observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        observation = next_observation

    observations = squeeze_array_envs(jnp.array(observations))
    next_observations = squeeze_array_envs(jnp.array(next_observations))
    actions = squeeze_array_envs(jnp.array(actions))
    rewards = squeeze_array_envs(jnp.array(rewards))
    dones = squeeze_array_envs(jnp.array(dones))

    trajectories = []
    for i in range(num_envs):
        terminations = (jnp.where(dones[i])[0] + 1).tolist()
        if len(terminations) == 0 or terminations[-1] != num_steps:
            terminations.append(num_steps)

        s = 0
        for t in terminations:
            trajectory = list(
                zip(
                    observations[i, s:t],
                    next_observations[i, s:t],
                    actions[i, s:t],
                    rewards[i, s:t],
                    dones[i, s:t],
                )
            )
            trajectories.append(trajectory)

            s = t

    return trajectories


def evaluate_policy(
    env: BaseEnv | SubprocVecEnv,
    model: BaseAlgorithm,
    n_eval_episodes: int = 128,
    return_episode_rewards: bool = False,
    progress_bar: bool = False,
):
    obs = env.reset()

    env_returns = [0 for _ in range(env.num_envs)]
    env_lengths = [0 for _ in range(env.num_envs)]

    episodes = 0
    episode_returns = []
    episode_lengths = []

    if progress_bar:
        bar = tqdm(total=n_eval_episodes)

    while True:
        actions = model.predict(obs)[0]
        obs, rewards, dones, _ = env.step(actions)

        for i, reward in enumerate(rewards):
            env_returns[i] += reward
            env_lengths[i] += 1
            if dones[i]:
                episodes += 1
                episode_returns.append(env_returns[i])
                episode_lengths.append(env_lengths[i])
                env_returns[i] = 0
                env_lengths[i] = 0

                if progress_bar:
                    bar.update(1)

            if episodes >= n_eval_episodes:
                if return_episode_rewards:
                    return episode_returns, episode_lengths

                return np.mean(episode_returns), np.std(episode_returns)
