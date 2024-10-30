import collections
from typing import List

import numpy as np
import torch
from torchrl.objectives import SACLoss
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.data import Bounded
from torchrl.envs.utils import RandomPolicy
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from gymnasium.spaces import Box

from diff_trans.envs.gym import BaseEnv

from .models import ActorNet, QValueNet, ValueNet, ClassifierNet


DarcLossInfo = collections.namedtuple(
    "DarcLossInfo",
    (
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "sa_classifier_loss",
        "sas_classifier_loss",
    ),
)


class DarcAgent:
    """An agent that implements the DARC algorithm."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        actor_net: ActorNet,
        q_value_net: QValueNet,
        value_net: ValueNet,
        classifier_net: ClassifierNet,
        actor_optimizer: torch.optim.Optimizer,
        q_value_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        classifier_optimizer: torch.optim.Optimizer,
        classifier_loss_weight: float = 1.0,
        use_importance_weights: bool = False,
        unnormalized_delta_r: bool = False,
        **kwargs,
    ):
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        self.classifier_loss_weight = classifier_loss_weight
        self.use_importance_weights = use_importance_weights
        self.unnormalized_delta_r = unnormalized_delta_r

        self.classifier_net = classifier_net
        self.classifier_optimizer = classifier_optimizer

        self.actor_net = actor_net
        self.q_value_net = q_value_net
        self.value_net = value_net

        self.actor_optimizer = actor_optimizer
        self.q_value_optimizer = q_value_optimizer
        self.value_optimizer = value_optimizer

        action_spec = Bounded(
            shape=torch.Size([self.action_dim]),
            low=torch.tensor(action_space.low),
            high=torch.tensor(action_space.high),
        )

        self.actor = ProbabilisticActor(
            module=TensorDictModule(
                self.actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
            ),
            spec=action_spec,
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
        )
        self.q_value = ValueOperator(
            module=self.q_value_net,
            in_keys=["observation", "action"],
        )
        self.value = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
        )
        self.sac_loss = SACLoss(
            actor_network=self.actor,
            qvalue_network=self.q_value,
            value_network=self.value,
            **kwargs,
        )

        self.random_actor = RandomPolicy(action_spec)

    @property
    def policy(self):
        return self.actor

    @property
    def random_policy(self):
        return self.random_actor

    def collect_experience(self, env: BaseEnv, num_steps: int):
        raise NotImplementedError

    def train(self, experience, weights, real_experience=None):
        raise NotImplementedError

    def _experience_to_sas(self, experience):
        raise NotImplementedError

    def _classifier_loss(self, experience, real_experience):
        raise NotImplementedError

    def critic_loss(
        self,
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn,
        gamma=1.0,
        reward_scale_factor=1.0,
        weights=None,
        training=False,
        delta_r_scale=1.0,
        delta_r_warmup=0,
    ):
        raise NotImplementedError

    def _check_train_argspec(self, kwargs):
        """Overwrite to avoid checking that real_experience has right dtype."""
        del kwargs
        return


class SimpleCollector:
    """A simple collector that collects experience from an environment."""

    def __init__(self, env: BaseEnv):
        self.env = env

        self.observations = env.reset()
        self.batch_size = self.observations.shape[0]

    def get_actions(self, policy, observations: np.ndarray) -> np.ndarray:
        observations_dict = TensorDict(
            {"observation": torch.tensor(observations)},
            [
                self.batch_size,
            ],
        )
        actions_dict = policy(observations_dict)

        return actions_dict["action"].detach().numpy()

    def collect(self, policy, num_steps: int) -> TensorDict:
        acc_observations = []
        acc_actions = []
        acc_dones = []
        acc_rewards = []
        acc_next_observations = []

        observations = self.observations
        for _ in range(num_steps):
            actions = self.get_actions(policy, self.observations)
            next_observations, rewards, dones, _ = self.env.step(actions)

            acc_observations.append(observations)
            acc_actions.append(actions)
            acc_dones.append(dones)
            acc_rewards.append(rewards)
            acc_next_observations.append(next_observations)

            observations = next_observations

        self.observations = observations

        observations = np.concatenate(acc_observations, axis=0)
        actions = np.concatenate(acc_actions, axis=0)
        dones = np.concatenate(acc_dones, axis=0)
        rewards = np.concatenate(acc_rewards, axis=0)
        next_observations = np.concatenate(acc_next_observations, axis=0)

        return TensorDict(
            {
                "observation": observations,
                "action": actions,
                ("next", "done"): dones,
                ("next", "terminated"): np.zeros_like(dones, dtype=bool),
                ("next", "reward"): rewards,
                ("next", "observation"): next_observations,
            },
            batch_size=[
                self.batch_size * num_steps,
            ],
        )
