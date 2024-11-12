import os
from typing import List


def get_exp_file_levels(root: str, filepath: str) -> List[str]:
    levels = []
    while True:
        filepath, level = os.path.split(filepath)
        level = os.path.splitext(level)[0]
        if level == root:
            break

        levels.append(level)

    levels.reverse()
    return levels


def get_exp_module_name(root: str, filepath: str) -> str:
    exp_levels = get_exp_file_levels(root, filepath)
    return "experiments." + ".".join(exp_levels)