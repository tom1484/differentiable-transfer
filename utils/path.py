import os
from typing import List


def get_exp_file_levels(root: str, filepath: str) -> str:
    levels = []
    while True:
        filepath, level = os.path.split(filepath)
        level = os.path.splitext(level)[0]
        if level == root:
            break

        levels.append(level)

    levels.reverse()
    return levels


def create_exp_dirs(
    root: str, exp_levels: List[str], exp_name: str, override: bool
) -> tuple[str, str]:
    exp_levels.append(exp_name)

    logs_dir = os.path.join(root, "logs", *exp_levels)
    models_dir = os.path.join(root, "models", *exp_levels)

    os.makedirs(logs_dir, exist_ok=override)
    os.makedirs(models_dir, exist_ok=override)

    return logs_dir, models_dir
