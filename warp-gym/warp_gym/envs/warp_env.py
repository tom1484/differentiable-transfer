from typing import Optional
import numpy as np


class WarpEnv:
    def __init__(self, fps: int, sim_substeps: int) -> None:
        self.fps = fps
        self.sim_substeps = sim_substeps

        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / sim_substeps

        self.sim_tick = 0
        self.sim_time = 0.0

    @property
    def state(self):
        pass

    @state.setter
    def state(self, value):
        pass

    def reset(self):
        pass

    def forward(self, control: np.ndarray):
        pass

    def step(self, control: Optional[np.ndarray] = None):
        pass

    def close(self):
        pass

    def render(self, path: str, scaling=1.0):
        pass
