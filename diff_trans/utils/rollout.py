from jax import numpy as jnp
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from ..envs.wrapped import BaseEnv


def squeeze_array_envs(array: jnp.ndarray):
    num_dims = len(array.shape)
    return jnp.transpose(array, (1, 0, *list(range(2, num_dims))))


def rollout_transitions(env: BaseEnv, model, num_transitions=100):
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


def evaluate_policy(env: BaseEnv, model: BaseAlgorithm, num_episodes=128):
    obs = env.reset()
    returns = []
    acc_rewards = [0 for _ in range(env.num_envs)]

    while True:
        actions = model.predict(obs)[0]
        obs, rewards, dones, _ = env.step(actions)
        
        for i, reward in enumerate(rewards):
            acc_rewards[i] += reward
            if dones[i]:
                returns.append(acc_rewards[i])
                acc_rewards[i] = 0
        
        if len(returns) >= num_episodes:
            break
    
    returns = np.array(returns[:num_episodes])
    return returns.mean(), returns.std()