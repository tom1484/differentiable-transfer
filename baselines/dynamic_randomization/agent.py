import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .actor import Actor
from .critic import Critic
from .noise import OrnsteinUhlenbeckActionNoise

from diff_trans.envs.wrapped import BaseEnv


MAX_STEPS = 50
TAU = 5e-3
LEARNING_RATE = 1e-3


class Agent:
    def __init__(self, env: BaseEnv, batch_size: int):
        self._dummy_env = env
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sum_writer = SummaryWriter("logs/")

        # Hardcoded for now
        self._dim_env = env.num_env
        self._dim_state = env.observation_space.shape[0]
        self._dim_action = env.action_space.shape[0]
        self._batch_size = batch_size

        # agent noise
        self._action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._dim_action))

        self._actor = Actor(
            self._dim_state,
            self._dim_action,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._batch_size
        ).to(self._device)

        self._critic = Critic(
            self._dim_state,
            self._dim_action,
            self._dim_env,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._actor.parameters(),
            self._sum_writer
        ).to(self._device)

        self._actor_target = Actor(
            self._dim_state,
            self._dim_action,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._batch_size
        ).to(self._device)

        self._critic_target = Critic(
            self._dim_state,
            self._dim_action,
            self._dim_env,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._actor.parameters(),
            self._sum_writer
        ).to(self._device)

        self._actor.initialize_target_network(self._actor_target)
        self._critic.initialize_target_network(self._critic_target)

        # training monitoring
        self._success_rate = torch.tensor(0.0, device=self._device)
        self._python_success_rate = torch.tensor(0.0, device=self._device)

    def get_dim_state(self):
        return self._dim_state

    def get_dim_action(self):
        return self._dim_action

    def get_dim_env(self):
        return self._dim_env

    def evaluate_actor(self, actor_predict, obs, history):
        assert history.shape[0] == MAX_STEPS, "history must be of size MAX_STEPS"
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        history = torch.tensor(history, dtype=torch.float32, device=self._device).unsqueeze(0)
        return actor_predict(obs, history)

    def evaluate_actor_batch(self, actor_predict, obs, history):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        history = torch.tensor(history, dtype=torch.float32, device=self._device)
        return actor_predict(obs, history)

    def evaluate_critic(self, critic_predict, obs, action, history, env):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self._device).unsqueeze(0)
        history = torch.tensor(history, dtype=torch.float32, device=self._device).unsqueeze(0)
        env = torch.tensor(env, dtype=torch.float32, device=self._device).unsqueeze(0)
        return critic_predict(env, obs, action, history)

    def evaluate_critic_batch(self, critic_predict, obs, action, history, env):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        action = torch.tensor(action, dtype=torch.float32, device=self._device)
        history = torch.tensor(history, dtype=torch.float32, device=self._device)
        env = torch.tensor(env, dtype=torch.float32, device=self._device)
        return critic_predict(env, obs, action, history)

    def train_critic(self, obs, action, history, env, predicted_q_value):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        action = torch.tensor(action, dtype=torch.float32, device=self._device)
        history = torch.tensor(history, dtype=torch.float32, device=self._device)
        env = torch.tensor(env, dtype=torch.float32, device=self._device)
        predicted_q_value = torch.tensor(predicted_q_value, dtype=torch.float32, device=self._device)
        return self._critic.train_critic(env, obs, action, history, predicted_q_value)

    def train_actor(self, obs, history, a_gradient):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        history = torch.tensor(history, dtype=torch.float32, device=self._device)
        a_gradient = torch.tensor(a_gradient, dtype=torch.float32, device=self._device)
        return self._actor.train_network(obs, history, a_gradient)

    def action_gradients_critic(self, obs, action, history, env):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        action = torch.tensor(action, dtype=torch.float32, device=self._device)
        history = torch.tensor(history, dtype=torch.float32, device=self._device)
        env = torch.tensor(env, dtype=torch.float32, device=self._device)
        return self._critic.action_gradients(env, obs, action, history)

    def update_target_actor(self):
        self._actor.update_target_network(self._actor_target)

    def update_target_critic(self):
        self._critic.update_target_network(self._critic_target)

    def action_noise(self):
        return self._action_noise()

    def update_success(self, success_rate, step):
        self._python_success_rate = torch.tensor(success_rate, dtype=torch.float32, device=self._device)
        self._success_rate = self._python_success_rate
        self._sum_writer.add_scalar("success_rate", self._success_rate.item(), step)

    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self._actor.state_dict(),
            'critic_state_dict': self._critic.state_dict(),
            'actor_target_state_dict': self._actor_target.state_dict(),
            'critic_target_state_dict': self._critic_target.state_dict(),
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self._actor.load_state_dict(checkpoint['actor_state_dict'])
        self._critic.load_state_dict(checkpoint['critic_state_dict'])
        self._actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self._critic_target.load_state_dict(checkpoint['critic_target_state_dict'])