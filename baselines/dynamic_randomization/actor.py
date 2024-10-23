from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from diff_trans.envs.gym import BaseEnv

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
        super().__init__()

        self._dim_state = dim_state
        self._dim_action = dim_action
        self._action_bound = env.action_space.high
        self._tau = tau
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        # Define the network
        self.ff_branch = nn.Sequential(nn.Linear(dim_state, UNITS), nn.ReLU())
        self.recurrent_branch = nn.LSTMCell(
            input_size=dim_state + dim_action, hidden_size=UNITS
        )
        self.rb_hidden = None
        self.rb_cell = None

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

    def reset_lstm_hidden_state(self, batch_size: int):
        device = next(self.parameters()).device
        self.rb_hidden = torch.zeros(batch_size, UNITS, device=device)
        self.rb_cell = torch.zeros(batch_size, UNITS, device=device)

    def forward(
        self,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ff_out = self.ff_branch(state)

        x = torch.cat([state, action_old], dim=1)

        if rb_state is None:
            rb_hidden, rb_cell = self.rb_hidden, self.rb_cell
            rb_hidden, rb_cell = self.recurrent_branch(x, (rb_hidden, rb_cell))
            self.rb_hidden, self.rb_cell = rb_hidden, rb_cell
        else:
            rb_hidden, rb_cell = rb_state
            rb_hidden, rb_cell = self.recurrent_branch(x, (rb_hidden, rb_cell))

        recurrent_out = rb_hidden

        merged_out = self.merged_branch(torch.cat([ff_out, recurrent_out], dim=1))
        scaled_out = merged_out * torch.tensor(
            self._action_bound, device=merged_out.device
        )
        return scaled_out, (rb_hidden, rb_cell)

    def train_network(
        self,
        state: torch.Tensor,
        action_old: torch.Tensor,
        a_gradient: torch.Tensor,
    ) -> None:
        self.optimizer.zero_grad()
        predictions = self.forward(state, action_old)
        loss = -torch.mean(
            predictions * a_gradient
        )  # Assuming a_gradient is the gradient from the critic
        loss.backward()
        self.optimizer.step()

    def predict(
        self,
        state: torch.Tensor,
        action_old: torch.Tensor,
        rb_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            return self.forward(state, action_old, rb_state)

    def update_target_network(self, target_actor: nn.Module) -> None:
        for target_param, param in zip(target_actor.parameters(), self.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1.0 - self._tau) * target_param.data
            )

    def initialize_target_network(self, target_actor: nn.Module) -> None:
        for target_param, param in zip(target_actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)


if __name__ == "__main__":
    from diff_trans.envs.gym import get_env
    from experiments.env import set_env_vars

    set_env_vars(jax_debug_nans=True)

    Env = get_env("InvertedPendulum-v5")
    env = Env()

    actor = Actor(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env,
        0.005,
        0.0001,
        32,
    )

    obs = env.reset()[0]
    # action = env.action_space.sample()
    action = torch.zeros(env.action_space.shape)

    print(actor.predict(torch.tensor(obs), action))
