from typing import List, Tuple

import torch
import torch.nn as nn
from torchrl.modules.distributions import NormalParamExtractor


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

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa_input = torch.cat([state, action], dim=-1)
        sas_input = torch.cat([state, action, next_state], dim=-1)

        if self.input_noise > 0:
            sa_input = sa_input + torch.randn_like(sa_input) * self.input_noise
            sas_input = sas_input + torch.randn_like(sas_input) * self.input_noise

        sa_logits = self.sa_classifier(sa_input)
        sas_logits = self.sas_classifier(sas_input)

        sa_probs = torch.softmax(sa_logits, dim=-1)
        sas_probs = torch.softmax(sas_logits, dim=-1)

        return sa_probs, sas_probs


# class ActorNet(nn.Module):
#     def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int]):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(observation_dim, hidden_dims[0]),
#             nn.Tanh(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.Tanh(),
#             nn.Linear(hidden_dims[1], action_dim * 2),
#         )
#         self.param_extractor = NormalParamExtractor()

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         return self.param_extractor(self.layers(state))


class ActorNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * 2),
        )
        self.normal_extractor = NormalParamExtractor()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.normal_extractor(self.layers(state))
        std = log_std.exp()

        return mu, std


class QValueNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)

        return self.layers(state_action)


class ValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
