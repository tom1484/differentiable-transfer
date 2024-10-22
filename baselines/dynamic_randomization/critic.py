import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from diff_trans.envs.wrapped import BaseEnv

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

        self.recurrent_branch = nn.LSTM(
            input_size=dim_action + dim_state, hidden_size=UNITS, batch_first=True
        )

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

    def forward(
        self,
        input_env: torch.Tensor,
        input_action: torch.Tensor,
        input_state: torch.Tensor,
        input_history: torch.Tensor,
    ) -> torch.Tensor:
        # Forward pass through the feedforward branch
        ff_input = torch.cat((input_env, input_action, input_state), dim=1)
        ff_out = self.ff_branch(ff_input)

        # Forward pass through the recurrent branch
        _, (recurrent_out, _) = self.recurrent_branch(input_history)
        recurrent_out = recurrent_out[-1]  # Get the last output of LSTM

        # Merge branches
        merged_input = torch.cat((ff_out, recurrent_out), dim=1)
        out = self.merged_branch(merged_input)
        return out

    def train_critic(
        self,
        input_env: torch.Tensor,
        input_state: torch.Tensor,
        input_action: torch.Tensor,
        input_history: torch.Tensor,
        predicted_q_value: torch.Tensor,
    ):
        self.optimizer.zero_grad()
        net_out = self.forward(input_env, input_action, input_state, input_history)
        loss = nn.MSELoss()(net_out, predicted_q_value)
        loss.backward()
        self.optimizer.step()

        # Log summaries
        self._sum_writer.add_scalar("loss", loss.item())
        return net_out

    def predict(
        self,
        input_env: torch.Tensor,
        input_state: torch.Tensor,
        input_action: torch.Tensor,
        input_history: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(input_env, input_action, input_state, input_history)

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
        input_env: torch.Tensor,
        input_state: torch.Tensor,
        input_action: torch.Tensor,
        input_history: torch.Tensor,
    ) -> torch.Tensor:
        # Enable gradient calculation
        input_action.requires_grad = True

        # Forward pass
        q_values = self.forward(input_env, input_action, input_state, input_history)

        # Compute gradients of q_values with respect to input_action
        q_values.backward(torch.ones_like(q_values))

        # Return the gradients
        return input_action.grad
