import torch
import torch.nn as nn
import torch.optim as optim

from diff_trans.envs.wrapped import BaseEnv

UNITS = 128
MAX_STEPS = 50


class Actor(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        env: BaseEnv,
        tau: float,
        learning_rate: float,
        batch_size: int,
    ):
        super(Actor, self).__init__()
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._action_bound = env.action_space.high
        self._tau = tau
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        # Define the network
        self.ff_branch = nn.Sequential(nn.Linear(dim_state, UNITS), nn.ReLU())
        self.recurrent_branch = nn.LSTM(
            input_size=dim_state + dim_action, hidden_size=UNITS, batch_first=True
        )
        self.merged_branch = nn.Sequential(
            nn.Linear(UNITS * 2, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, dim_action),
            nn.Tanh(),
        )

        # Initialize weights
        self.merged_branch[-2].weight.data.uniform_(-0.003, 0.003)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        ff_out = self.ff_branch(state)
        recurrent_out, _ = self.recurrent_branch(memory)
        recurrent_out = recurrent_out[:, -1, :]  # Take the last output of the LSTM
        merged_out = self.merged_branch(torch.cat([ff_out, recurrent_out], dim=-1))
        scaled_out = merged_out * torch.tensor(
            self._action_bound, device=merged_out.device
        )
        return scaled_out

    def train_network(
        self,
        input_state: torch.Tensor,
        input_history: torch.Tensor,
        a_gradient: torch.Tensor,
    ) -> None:
        self.optimizer.zero_grad()
        predictions = self.forward(input_state, input_history)
        loss = -torch.mean(
            predictions * a_gradient
        )  # Assuming a_gradient is the gradient from the critic
        loss.backward()
        self.optimizer.step()

    def predict(
        self, input_state: torch.Tensor, input_history: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(input_state, input_history)

    def update_target_network(self, target_actor: nn.Module) -> None:
        for target_param, param in zip(target_actor.parameters(), self.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )

    def initialize_target_network(self, target_actor: nn.Module) -> None:
        for target_param, param in zip(target_actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
