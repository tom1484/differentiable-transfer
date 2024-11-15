import copy
from dataclasses import dataclass, field


def default_dict(**kwargs):
    return field(default_factory=lambda: copy.copy(kwargs))


def default_list(*args):
    return field(default_factory=lambda: copy.copy(args))
