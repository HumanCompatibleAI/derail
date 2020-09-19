import gym
from gym.spaces import Box
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import (
    grid_transition_fn,
    LightweightRLModel,
)


class ProcGoalEnv(BaseEnv):
    def __init__(self, bounds=100, distance=10):
        self.bounds = bounds
        self.distance = distance

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,))

        super().__init__(num_actions=5)

    def sample_initial_state(self):
        pos = self.np_random.randint(low=-self.bounds, high=self.bounds, size=(2,))

        x_dist = self.np_random.randint(self.distance)
        y_dist = self.distance - x_dist
        random_signs = 2 * self.np_random.randint(2, size=2) - 1
        goal = pos + random_signs * (x_dist, y_dist)

        return np.concatenate([pos, goal])

    def reward_fn(self, state, act, next_state):
        return (-1) * np.sum(np.abs(next_state[2:] - next_state[:2]))

    def transition_fn(self, state, action):
        pos, goal = state[:2], state[2:]
        next_pos = grid_transition_fn(pos, action)
        return np.concatenate([next_pos, goal])


_horizon_v0 = 20

gym.register(
    id=f"seals/ProcGoal-v0",
    entry_point=f"derail.envs:ProcGoalEnv",
    max_episode_steps=_horizon_v0,
)
