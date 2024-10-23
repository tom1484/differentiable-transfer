from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from baselines.dynamic_randomization.environment import RandomizedEnvironment
from baselines.dynamic_randomization.agent import Agent
from baselines.dynamic_randomization.memory import EpisodicMemory
from diff_trans.envs.wrapped import get_env

EPISODES = 2

env_name = "InvertedPendulum-v5"
Env = get_env(env_name)
env = Env(max_episode_steps=100)

# Program hyperparameters
TESTING_INTERVAL = 10  # number of updates between two evaluation of the policy
TESTING_ROLLOUTS = 100  # number of rollouts performed to evaluate the current policy

# Algorithm hyperparameters
BATCH_SIZE = 32
CAPACITY = 1000000
MAX_STEPS = 100  # WARNING: defined in multiple files...
GAMMA = 0.99

# Initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(env, BATCH_SIZE)

# Initialize the environment sampler
parameter_mask = np.ones(8, dtype=bool)
parameter_mask[5] = False
randomized_environment = RandomizedEnvironment(env_name, parameter_mask=parameter_mask)

# Initialize the replay buffer
# replay_buffer = ReplayBuffer(BUFFER_SIZE)
# num_updates = 0
memory = EpisodicMemory(CAPACITY, MAX_STEPS)


total_steps = 0
update_steps = 0

pbar = tqdm(total=CAPACITY, desc="Total Steps")
while total_steps < CAPACITY:
    # print(total_steps)

    # generate an environment
    randomized_environment.sample_env()
    env, env_params = randomized_environment.get_env()
    env_params = np.array(env_params)

    # reset the environment
    obs = env.reset()
    action_old = torch.zeros(
        env.action_space.shape, dtype=torch.float32, device="cuda"
    ).unsqueeze(0)

    done = False

    agent.reset_lstm_hidden_state(1)
    # rollout the  whole episode
    while not done:
        action, _ = agent.predict_action_single(
            agent.actor.predict,
            torch.tensor(obs[0], dtype=torch.float32, device="cuda"),
            action_old[0],
        )

        noise = agent.action_noise()
        action = action + torch.tensor(noise, dtype=torch.float32, device="cuda")
        # action = env.action_space.sample()

        np_action = action.cpu().numpy()
        new_obs, step_reward, done, info = env.step(np_action)
        memory.append(
            env_params, obs[0], np_action[0], step_reward[0], new_obs[0], done[0]
        )
        total_steps += 1

        obs = new_obs
        action_old = action

    env.close()

    if len(memory.memory) >= BATCH_SIZE:
        experiences = memory.sample(BATCH_SIZE)

        state_batches = np.zeros(
            [BATCH_SIZE, MAX_STEPS, agent.get_dim_state()], dtype=np.float32
        )
        state_next_batches = np.zeros(
            [BATCH_SIZE, MAX_STEPS, agent.get_dim_state()], dtype=np.float32
        )

        action_batches = np.zeros(
            [BATCH_SIZE, MAX_STEPS, agent.get_dim_action()], dtype=np.float32
        )
        action_old_batches = np.zeros(
            [BATCH_SIZE, MAX_STEPS, agent.get_dim_action()], dtype=np.float32
        )

        reward_batches = np.zeros([BATCH_SIZE, MAX_STEPS], dtype=np.float32)
        done_batches = np.zeros([BATCH_SIZE, MAX_STEPS], dtype=np.float32)

        env_params_batches = np.zeros(
            [BATCH_SIZE, MAX_STEPS, agent.get_dim_env()], dtype=np.float32
        )

        for t in range(len(experiences)):
            for b, transition in enumerate(experiences[t]):
                env_params_batches[b, t] = transition.env
                state_batches[b, t] = transition.state0
                action_batches[b, t] = transition.action
                reward_batches[b, t] = transition.reward
                state_next_batches[b, t] = transition.state1
                done_batches[b, t] = transition.terminal

            if t > 0:
                for b, transition in enumerate(experiences[t - 1]):
                    action_old_batches[b, t] = transition.action

        for t in range(len(experiences) - 1):
            action_old_batches[:, t + 1] *= (1 - done_batches[:, t])[:, None]

        agent.reset_lstm_hidden_state()

        actor_state = (agent.actor.rb_hidden, agent.actor.rb_cell)
        critic_state = (agent.critic.rb_hidden, agent.critic.rb_cell)
        actor_target_state = (agent.actor_target.rb_hidden, agent.actor_target.rb_cell)
        critic_target_state = (
            agent.critic_target.rb_hidden,
            agent.critic_target.rb_cell,
        )

        value_loss_total = 0
        policy_loss_total = 0

        for t in range(len(experiences)):
            env_params = torch.tensor(env_params_batches[:, t], device="cuda")
            state = torch.tensor(state_batches[:, t], device="cuda")
            action = torch.tensor(action_batches[:, t], device="cuda")
            reward = torch.tensor(reward_batches[:, t, None], device="cuda")
            state_next = torch.tensor(state_next_batches[:, t], device="cuda")
            action_old = torch.tensor(action_old_batches[:, t], device="cuda")
            done = torch.tensor(done_batches[:, t, None], device="cuda")

            predicted_target_action, actor_target_state = agent.predict_action(
                agent.actor_target.predict, state_next, action, actor_target_state
            )
            target_q_value, critic_target_state = agent.predict_q(
                agent.critic_target.predict,
                env_params,
                predicted_target_action,
                state_next,
                action,
                critic_target_state,
            )

            current_q_value, critic_state = agent.predict_q(
                agent.critic, env_params, action, state, action_old, critic_state
            )
            diff_q_value = (reward + GAMMA * target_q_value - current_q_value).detach()

            value_loss_total += diff_q_value * current_q_value / len(experiences)

            # Calculate action gradients
            predicted_action, _ = agent.predict_action(
                agent.actor.predict, state, action_old, actor_state
            )
            predicted_action.requires_grad = True
            predicted_q_value, _ = agent.predict_q(
                agent.critic,
                env_params,
                predicted_action,
                state,
                action_old,
                critic_state,
            )
            agent.critic.zero_grad()
            predicted_q_value.backward(
                torch.ones_like(predicted_q_value),
                retain_graph=True,
                inputs=[predicted_action],
            )
            action_grad = predicted_action.grad

            # Calculate policy loss
            prediction, actor_state = agent.predict_action(
                agent.actor, state, action_old, actor_state
            )
            policy_loss_total += action_grad * prediction / len(experiences)

        agent.critic.optimizer.zero_grad()
        value_grad = value_loss_total.sum().backward()
        agent.critic.optimizer.step()

        agent.actor.optimizer.zero_grad()
        policy_grad = policy_loss_total.sum().backward()
        agent.actor.optimizer.step()

        update_steps += 1

        if update_steps % TESTING_INTERVAL == 0:
            total_episodic_reward = 0
            for ep in range(TESTING_ROLLOUTS):
                # generate an environment
                randomized_environment.sample_env()
                env, env_params = randomized_environment.get_env()

                # reset the environment
                obs = env.reset()
                action_old = torch.zeros(
                    env.action_space.shape, dtype=torch.float32, device="cuda"
                ).unsqueeze(0)

                done = False
                episodic_reward = 0

                agent.reset_lstm_hidden_state(1)
                # rollout the  whole episode
                while not done:
                    action, _ = agent.predict_action_single(
                        agent.actor.predict,
                        torch.tensor(obs[0], dtype=torch.float32, device="cuda"),
                        action_old[0],
                    )

                    np_action = action.cpu().numpy()
                    new_obs, step_reward, done, info = env.step(np_action)
                    episodic_reward += step_reward

                    obs = new_obs
                    action_old = action

                env.close()
                total_episodic_reward += episodic_reward

            print(
                f"Average episodic reward: {total_episodic_reward / TESTING_ROLLOUTS}"
            )
            agent.save_model(
                f"baselines/dynamic_randomization/checkpoints/model_{update_steps}.pth"
            )

    pbar.n = total_steps
    pbar.refresh()
