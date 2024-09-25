import os
from typing import Dict, Type
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm


ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": stable_baselines3.PPO,
    "A2C": stable_baselines3.A2C,
    "DQN": stable_baselines3.DQN,
    "DDPG": stable_baselines3.DDPG,
    "SAC": stable_baselines3.SAC,
    "TD3": stable_baselines3.TD3,
}