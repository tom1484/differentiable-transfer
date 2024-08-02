import os
from typing import List


def create_session(levels: List[str]):
    name = "-".join(levels)
    os.system(f"tmux new-session -d -s {name}")