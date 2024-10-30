from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        input_noise: float = 0.1,
    ):
        super().__init__()

        self.input_noise = input_noise

        self.sa_classifier = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.sas_classifier = nn.Sequential(
            nn.Linear(2 * observation_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.s_dim = observation_dim
        self.a_dim = action_dim
        self.sas_dim = 2 * self.s_dim + self.a_dim
        self.sa_dim = self.sas_dim - self.s_dim

    def forward(self, sas_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa_input = sas_input[:, : -self.s_dim]

        if self.input_noise > 0:
            sa_input = sa_input + torch.randn_like(sa_input) * self.input_noise
            sas_input = sas_input + torch.randn_like(sas_input) * self.input_noise

        sa_logits = self.sa_classifier(sa_input)
        sas_logits = self.sas_classifier(sas_input)

        sa_probs = torch.softmax(sa_logits, dim=-1)
        sas_probs = torch.softmax(sas_logits, dim=-1)

        return sa_probs, sas_probs


class ActorNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.fc1 = nn.Linear(observation_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)

        return mean, log_std


class QValueNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.fc1 = nn.Linear(observation_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q_value = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)

        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))

        return self.q_value(x)


class ValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.fc1 = nn.Linear(observation_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.value = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.value(x)
