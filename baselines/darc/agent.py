from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm

from torchrl.objectives import SACLoss
from torchrl.objectives.utils import SoftUpdate
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.data import Bounded
from torchrl.envs.utils import RandomPolicy
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from gymnasium.spaces import Box

from diff_trans.envs.gym_wrapper import BaseEnv

from .models import ActorNet, QValueNet, ValueNet, ClassifierNet


def check_grad_magnitude(params: List[torch.nn.Parameter]):
    acc = 0
    count = 0
    for param in params:
        if param.grad is not None:
            acc += param.grad.norm().item()
            count += 1

    if count > 0:
        print(f"Total grad magnitude: {acc / count}")
    else:
        print("No gradients found")


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
        classifier_optimizer: torch.optim.Optimizer,
        learning_rate: float,
        classifier_loss_weight: float = 1.0,
        use_importance_weights: bool = False,
        unnormalized_delta_r: bool = False,
        target_update_tau: float = 0.01,
        target_update_period: int = 1,
        **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        self.classifier_loss_weight = classifier_loss_weight
        self.use_importance_weights = use_importance_weights
        self.unnormalized_delta_r = unnormalized_delta_r

        self.classifier_net = classifier_net.cuda()
        self.classifier_optimizer = classifier_optimizer

        self.classifier = TensorDictModule(
            self.classifier_net,
            in_keys=["observation", "action", ("next", "observation")],
            out_keys=["sa_logits", "sas_logits"],
        ).cuda()

        self.actor_net = actor_net.cuda()
        self.q_value_net = q_value_net.cuda()
        self.value_net = value_net.cuda()

        action_spec = Bounded(
            shape=torch.Size([self.action_dim]),
            low=torch.tensor(action_space.low),
            high=torch.tensor(action_space.high),
        )

        self.actor = ProbabilisticActor(
            module=SafeModule(
                self.actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
            ),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dict(
                low=torch.tensor(action_space.low),
                high=torch.tensor(action_space.high),
            ),
        ).cuda()
        self.q_value = ValueOperator(
            module=self.q_value_net,
            in_keys=["observation", "action"],
        ).cuda()
        self.value = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
        ).cuda()
        self.sac_loss = SACLoss(
            actor_network=self.actor,
            qvalue_network=self.q_value,
            value_network=self.value,
            **kwargs,
        ).cuda()

        self.optimizer = Adam(self.sac_loss.parameters(), lr=learning_rate)

        self.target_updater = SoftUpdate(self.sac_loss, tau=target_update_tau)
        self.target_update_period = target_update_period
        self.num_updates = 0

    @property
    def policy(self):
        return self.sac_loss.actor_network

    def get_init_policy(self, batch_size: int):
        return RandomPolicy(
            Bounded(
                shape=torch.Size([batch_size, self.action_dim]),
                low=torch.tensor(self.action_space.low).repeat(batch_size, 1),
                high=torch.tensor(self.action_space.high).repeat(batch_size, 1),
            )
        )

    def update(
        self, batch: TensorDict, real_batch: Optional[TensorDict] = None
    ) -> TensorDict:
        batch = batch.cuda()

        if real_batch is not None:
            real_batch = real_batch.cuda()
            classifier_loss, sa_loss, sas_loss = self._classifier_loss(
                batch, real_batch
            )
            self.classifier_optimizer.zero_grad()
            classifier_loss.backward()
            self.classifier_optimizer.step()

            # Calculate reward modification
            batch = self._calculate_delta_r(batch)

        # Update SAC
        losses = self.sac_loss(batch)

        loss = (
            losses["loss_actor"]
            + losses["loss_qvalue"]
            + losses["loss_value"]
            + losses["entropy"]
            + losses["alpha"]
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.num_updates += 1
        if self.num_updates % self.target_update_period == 0:
            self.target_updater.step()

        # losses["loss_classifier"] = classifier_loss
        # losses["sa_classifier_loss"] = sa_loss
        # losses["sas_classifier_loss"] = sas_loss

        return losses

    def _classifier_loss(self, experience: TensorDict, real_experience: TensorDict):
        batch_size = real_experience["observation"].shape[0]
        all_experience = torch.cat([experience, real_experience], dim=0)
        y_true = torch.cat(
            [
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long),
            ],
            dim=0,
        )

        logits = self.classifier(all_experience)
        sa_loss = F.cross_entropy(logits["sa_logits"], y_true)
        sas_loss = F.cross_entropy(logits["sas_logits"], y_true)
        loss = self.classifier_loss_weight * (sa_loss + sas_loss)

        return loss, sa_loss, sas_loss

    def _calculate_delta_r(self, batch: TensorDict):
        with torch.no_grad():
            logits = self.classifier(batch)
            sa_logits, sas_logits = logits["sa_logits"], logits["sas_logits"]
            delta_r = (
                sas_logits[:, 1] - sas_logits[:, 0] - sa_logits[:, 1] + sa_logits[:, 0]
            )[:, None]

            batch["next", "reward"] += delta_r

        return batch

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "q_value": self.q_value_net.state_dict(),
                "value": self.value_net.state_dict(),
                "classifier": self.classifier_net.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(
        path: str,
        actor_net: ActorNet,
        q_value_net: QValueNet,
        value_net: ValueNet,
        classifier_net: ClassifierNet,
    ):
        checkpoint = torch.load(path)
        actor_net.load_state_dict(checkpoint["actor"])
        q_value_net.load_state_dict(checkpoint["q_value"])
        value_net.load_state_dict(checkpoint["value"])
        classifier_net.load_state_dict(checkpoint["classifier"])


def predict(policy, observations: np.ndarray) -> np.ndarray:
    observations_dict = TensorDict(
        {"observation": observations},
        [
            observations.shape[0],
        ],
    ).cuda()
    actions_dict = policy(observations_dict)

    return actions_dict["action"].detach().cpu().numpy()


def evaluate_policy(env: BaseEnv, policy, num_episodes: int) -> float:
    ep_returns = []
    acc_returns = [0.0 for _ in range(env.num_envs)]

    observations = env.reset()
    with tqdm(total=num_episodes) as pbar:
        while len(ep_returns) < num_episodes:
            actions = predict(policy, observations)
            next_observations, rewards, dones, _ = env.step(actions)
            observations = next_observations

            for i, (reward, done) in enumerate(zip(rewards, dones)):
                acc_returns[i] += reward
                if done:
                    ep_returns.append(acc_returns[i])
                    acc_returns[i] = 0
                    pbar.update(1)

    return np.mean(ep_returns), np.std(ep_returns)


class SimpleCollector:
    """A simple collector that collects experience from an environment."""

    def __init__(self, env: BaseEnv):
        self.env = env

        self.observations = env.reset()
        self.batch_size = self.observations.shape[0]

    def collect(self, policy, num_steps: int, progress_bar: bool = False) -> TensorDict:
        acc_observations = []
        acc_actions = []
        acc_dones = []
        acc_rewards = []
        acc_next_observations = []

        observations = self.observations
        for _ in tqdm(range(num_steps), disable=not progress_bar):
            actions = predict(policy, observations)
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
        dones = np.concatenate(acc_dones, axis=0)[:, None]
        rewards = np.concatenate(acc_rewards, axis=0)[:, None]
        next_observations = np.concatenate(acc_next_observations, axis=0)

        if len(actions.shape) == 1:
            actions = actions[:, None]

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
