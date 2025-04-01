# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
sys.path.append(r'../')
import numpy as np
import torch


# env wrapper
class env_wrapper:
    def __init__(self, env, episode_length=1500000):
        self._env = env
        self.episode_length = episode_length
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        self.timesteps = 0
        obs = self._env.reset()
        return obs

    def step(self, action):
        # revise the correct action range
        obs, reward, done, info = self._env.step(action)
        # increase the timesteps
        self.timesteps += 1
        if self.timesteps >= self.episode_length:
            done = True
        return obs, reward, done, info

    def render(self):
        """
        to be Implemented during execute the demo
        """
        self._env.render()

    def seed(self, seed):
        """
        set environment seeds
        """
        self._env.seed(seed)


# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)

    @property
    def mean(self):
        return np.mean(self.buffer)

    # get the length of total episodes
    @property
    def num_episodes(self):
        return self._episode_length

