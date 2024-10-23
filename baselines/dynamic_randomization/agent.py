from typing import Optional, Tuple

import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .actor import Actor
from .critic import Critic
from .noise import OrnsteinUhlenbeckActionNoise

from diff_trans.envs.gym import BaseEnv


MAX_STEPS = 50
TAU = 5e-3
LEARNING_RATE = 1e-3


class Agent:
    def __init__(self, env: BaseEnv, batch_size: int):
        self._dummy_env = env
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sum_writer = SummaryWriter("logs/")

        # Hardcoded for now
        self._dim_env = env.diff_env.parameter_range.shape[1]
        self._dim_state = env.observation_space.shape[0]
        self._dim_action = env.action_space.shape[0]
        self._batch_size = batch_size

        # agent noise
        self._action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._dim_action))

        self.actor = Actor(
            self._dim_state,
            self._dim_action,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._batch_size,
        ).to(self._device)

        self.critic = Critic(
            self._dim_state,
            self._dim_action,
            self._dim_env,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self.actor.parameters(),
            self._sum_writer,
        ).to(self._device)

        self.actor_target = Actor(
            self._dim_state,
            self._dim_action,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self._batch_size,
        ).to(self._device)

        self.critic_target = Critic(
            self._dim_state,
            self._dim_action,
            self._dim_env,
            self._dummy_env,
            TAU,
            LEARNING_RATE,
            self.actor.parameters(),
            self._sum_writer,
        ).to(self._device)

        self.actor.initialize_target_network(self.actor_target)
        self.critic.initialize_target_network(self.critic_target)

        # training monitoring
        self._success_rate = torch.tensor(0.0, device=self._device)
        self._python_success_rate = torch.tensor(0.0, device=self._device)

    def reset_lstm_hidden_state(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self._batch_size
        self.actor_target.reset_lstm_hidden_state(batch_size)
        self.critic_target.reset_lstm_hidden_state(batch_size)
        self.actor.reset_lstm_hidden_state(batch_size)
        self.critic.reset_lstm_hidden_state(batch_size)

    def get_dim_state(self):
        return self._dim_state

    def get_dim_action(self):
        return self._dim_action

    def get_dim_env(self):
        return self._dim_env

    def predict_action_single(
        self, actor: Actor, state: torch.Tensor, action_old: torch.Tensor
    ) -> torch.Tensor:
        # state = torch.tensor(state, dtype=torch.float32, device=self._device)
        # action_old = torch.tensor(
        #     action_old, dtype=torch.float32, device=self._device
        # )

        return actor(state.unsqueeze(0), action_old.unsqueeze(0))

    def predict_action(
        self,
        actor: Actor,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # state = torch.tensor(state, dtype=torch.float32, device=self._device)
        # action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)

        return actor(state, action_old, rb_state)

    def predict_q_single(
        self,
        critic_predict: Critic,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
    ) -> torch.Tensor:
        # env_params = torch.tensor(env_params, dtype=torch.float32, device=self._device)
        # action = torch.tensor(action, dtype=torch.float32, device=self._device)
        # state = torch.tensor(state, dtype=torch.float32, device=self._device)
        # action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)

        return critic_predict(
            env_params.unsqueeze(0),
            action.unsqueeze(0),
            state.unsqueeze(0),
            action_old.unsqueeze(0),
        )

    def predict_q(
        self,
        critic_predict: Critic,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # env_params = torch.tensor(env_params, dtype=torch.float32, device=self._device)
        # action = torch.tensor(action, dtype=torch.float32, device=self._device)
        # state = torch.tensor(state, dtype=torch.float32, device=self._device)
        # action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)

        return critic_predict(
            env_params, action, state, action_old, rb_state
        )

    def train_critic(
        self,
        env_params: np.ndarray,
        action: np.ndarray,
        state: np.ndarray,
        action_old: np.ndarray,
        predicted_q_value: np.ndarray,
    ):
        env_params = torch.tensor(env_params, dtype=torch.float32, device=self._device)
        action = torch.tensor(action, dtype=torch.float32, device=self._device)
        state = torch.tensor(state, dtype=torch.float32, device=self._device)
        action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)
        predicted_q_value = torch.tensor(
            predicted_q_value, dtype=torch.float32, device=self._device
        )

        return self.critic.train_critic(
            env_params, action, state, action_old, predicted_q_value
        )

    def train_actor(
        self, state: np.ndarray, action_old: np.ndarray, a_gradient: np.ndarray
    ):
        state = torch.tensor(state, dtype=torch.float32, device=self._device)
        action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)
        a_gradient = torch.tensor(a_gradient, dtype=torch.float32, device=self._device)

        return self.actor.train_network(state, action_old, a_gradient)

    def action_gradients_critic(
        self,
        env_params: np.ndarray,
        action: np.ndarray,
        state: np.ndarray,
        action_old: np.ndarray,
    ):
        env_params = torch.tensor(env_params, dtype=torch.float32, device=self._device)
        action = torch.tensor(action, dtype=torch.float32, device=self._device)
        state = torch.tensor(state, dtype=torch.float32, device=self._device)
        action_old = torch.tensor(action_old, dtype=torch.float32, device=self._device)

        return self.critic.action_gradients(env_params, action, state, action_old)

    def update_target_actor(self):
        self.actor.update_target_network(self.actor_target)

    def update_target_critic(self):
        self.critic.update_target_network(self.critic_target)

    def action_noise(self):
        return self._action_noise()

    def update_success(self, success_rate, step):
        self._python_success_rate = torch.tensor(
            success_rate, dtype=torch.float32, device=self._device
        )
        self._success_rate = self._python_success_rate
        self._sum_writer.add_scalar("success_rate", self._success_rate.item(), step)

    def save_model(self, filename):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
            },
            filename,
        )

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
