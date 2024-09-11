import os
import json
from typing import Dict, List, Tuple


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


def create_exp_assets(
    root: str, exp_levels: List[str], exp_name: str, default_config_dict: Dict
) -> Tuple[bool, str, str]:
    configs_dir = os.path.join(root, "configs", *exp_levels)
    config_path = os.path.join(configs_dir, f"{exp_name}.json")
    os.makedirs(configs_dir, exist_ok=True)
    
    config_exists = os.path.exists(config_path)
    if not config_exists:
        config_file = open(config_path, "w")
        json.dump(default_config_dict, config_file)
        config_file.close()

    models_dir = os.path.join(root, "models", *exp_levels, exp_name)
    os.makedirs(models_dir, exist_ok=True)

    return not config_exists, config_path, models_dir
