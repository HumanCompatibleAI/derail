import itertools

import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from dr_seals.envs.base_env import BaseEnv


class BranchingEnv(BaseEnv):
    def __init__(self, branch_factor=3, length=10, shaping_term=0):
        self.branch_factor = branch_factor
        self.shaping_term = shaping_term

        nS = 1 + branch_factor * length
        nA = branch_factor

        super().__init__(num_states=nS, num_actions=nA)

        self.transition_matrix = np.zeros((nS, nA, nS))
        for ob, act in itertools.product(range(nS), range(nA)):
            self.transition_matrix[ob, act, self._get_next(ob, act)] = 1.0

    def _get_next(self, state, action):
        b = self.branch_factor
        n = self.observation_space.n

        if state % b == 0 and state != n - 1:
            return state + (action + 1)
        else:
            return state

    def reward_fn(self, state, act, next_state):
        num_states = self.observation_space.n
        goal_reward = int(next_state == num_states - 1)
        shaping_reward = self.shaping_term * (
            state % self.branch_factor == 0 and next_state % self.branch_factor == 0
        )
        return goal_reward + shaping_reward


_horizon_v0 = 10

gym.register(
    id=f"seals/Branching-v0",
    entry_point=f"dr_seals.envs:BranchingEnv",
    max_episode_steps=_horizon_v0,
)