from typing import Type

from ..base import BaseEnv


ENTRIES = {}


def register(name: str, entry: Type[BaseEnv]):
    ENTRIES[name] = entry


def get_env(name: str) -> Type[BaseEnv]:
    return ENTRIES[name]