from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from diff_trans.envs.gym_wrapper import BaseEnv

UNITS = 128
MAX_STEPS = 50


class Critic(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_env: int,
        env: BaseEnv,
        tau: float,
        learning_rate: float,
        num_actor_vars: int,
        writer: SummaryWriter,
    ):
        super(Critic, self).__init__()

        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_env = dim_env
        self._action_bound = env.action_space.high

        self._learning_rate = learning_rate
        self._tau = tau
        self._sum_writer = writer

        # Define the network
        self.ff_branch = nn.Sequential(
            nn.Linear(dim_env + dim_action + dim_state, UNITS), nn.ReLU()
        )
        self.recurrent_branch = nn.LSTMCell(
            input_size=dim_state + dim_action, hidden_size=UNITS
        )
        self.rb_hidden = None
        self.rb_cell = None

        self.merged_branch = nn.Sequential(
            nn.Linear(2 * UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, 1),
        )

        # Initialize weights
        self.merged_branch[-1].weight.data.uniform_(-0.003, 0.003)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)

    def reset_lstm_hidden_state(self, batch_size: int):
        device = next(self.parameters()).device
        self.rb_hidden = torch.zeros(batch_size, UNITS, device=device)
        self.rb_cell = torch.zeros(batch_size, UNITS, device=device)


    def forward(
        self,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Forward pass through the feedforward branch
        ff_input = torch.cat((env_params, action, state), dim=1)
        ff_out = self.ff_branch(ff_input)

        # Forward pass through the recurrent branch
        x = torch.cat([state, action_old], dim=1)

        if rb_state is None:
            rb_hidden, rb_cell = self.rb_hidden, self.rb_cell
            rb_hidden, rb_cell = self.recurrent_branch(x, (rb_hidden, rb_cell))
            self.rb_hidden, self.rb_cell = rb_hidden, rb_cell
        else:
            rb_hidden, rb_cell = rb_state
            rb_hidden, rb_cell = self.recurrent_branch(x, (rb_hidden, rb_cell))

        recurrent_out = self.rb_hidden

        # Merge branches
        merged_out = self.merged_branch(torch.cat((ff_out, recurrent_out), dim=1))

        return merged_out, (rb_hidden, rb_cell)

    def train_critic(
        self,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
        predicted_q_value: torch.Tensor,
    ):
        self.optimizer.zero_grad()
        net_out = self.forward(env_params, action, state, action_old)[0]
        loss = nn.MSELoss()(net_out, predicted_q_value)
        loss.backward()
        self.optimizer.step()

        # Log summaries
        self._sum_writer.add_scalar("loss", loss.item())
        return net_out

    def predict(
        self,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(env_params, action, state, action_old, rb_state)

    def update_target_network(self, target_network: nn.Module):
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )

    def initialize_target_network(self, target_network: nn.Module):
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

    def action_gradients(
        self,
        env_params: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        action_old: torch.Tensor,
    ) -> torch.Tensor:
        # Enable gradient calculation
        action.requires_grad = True

        # Forward pass
        q_values = self.forward(env_params, action, state, action_old)

        # Compute gradients of q_values with respect to input_action
        q_values.backward(torch.ones_like(q_values))

        # Return the gradients
        return action.grad
