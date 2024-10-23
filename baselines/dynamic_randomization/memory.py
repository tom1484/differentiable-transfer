from typing import Optional, List

# from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple("Experience", "env, state0, action, reward, state1, terminal")


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError(idx)
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    if hasattr(observation, "shape"):
        return np.zeros(observation.shape)
    elif hasattr(observation, "__iter__"):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.0


class Memory(object):
    def __init__(self, window_length: int, ignore_episode_boundaries: bool = False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size: int, batch_idxs: Optional[List[int]] = None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = (
                self.recent_terminals[current_idx - 1]
                if current_idx - 1 >= 0
                else False
            )
            if current_idx < 0 or (
                not self.ignore_episode_boundaries and current_terminal
            ):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            "window_length": self.window_length,
            "ignore_episode_boundaries": self.ignore_episode_boundaries,
        }
        return config


class EpisodicMemory(Memory):
    def __init__(self, capacity: int, max_episode_length: int, **kwargs):
        # super(EpisodicMemory, self).__init__(**kwargs)
        # Max number of transitions possible will be the memory capacity, could be much less
        self.max_episode_length = max_episode_length
        self.num_episodes = capacity // max_episode_length
        self.memory = RingBuffer(self.num_episodes)
        self.trajectory = []  # Temporal list of episode

    def append(
        self,
        env: np.ndarray,
        state0: np.ndarray,
        action: np.ndarray,
        reward: float,
        state1: np.ndarray,
        terminal: bool,
        training=True,
    ):
        self.trajectory.append(
            Experience(
                env=env,
                state0=state0,
                action=action,
                reward=reward,
                state1=state1,
                terminal=terminal,
            )
        )  #
        if len(self.trajectory) >= self.max_episode_length:
            self.memory.append(self.trajectory)
            self.trajectory = []

    def sample(self, batch_size: int, maxlen: int = 0):
        batch = [self.sample_trajectory(maxlen=maxlen) for _ in range(batch_size)]
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [
            trajectory[:minimum_size] for trajectory in batch
        ]  # Truncate trajectories
        return list(
            map(list, zip(*batch))
        )  # Transpose so that timesteps are packed together

    def sample_trajectory(self, maxlen: int):
        e = random.randrange(len(self.memory))
        mem = self.memory[e]
        T = len(mem)
        if T > 0:
            # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
            if maxlen > 0 and T > maxlen + 1:
                t = random.randrange(
                    T - maxlen - 1
                )  # Include next state after final "maxlen" state
                return mem[t : t + maxlen + 1]
            else:
                return mem

    def __len__(self):
        return sum(len(self.memory[idx]) for idx in range(len(self.memory)))
