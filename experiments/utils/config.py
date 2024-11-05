import copy
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


def default_dict(**kwargs):
    return field(default_factory=lambda: copy.copy(kwargs))
