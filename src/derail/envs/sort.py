import gym
from gym.spaces import MultiDiscrete, Box
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import LightweightRLModel


class SortEnv(BaseEnv):
    def __init__(self, length=4):
        self.length = length

        self.observation_space = Box(low=0, high=1.0, shape=(length,))
        self.action_space = MultiDiscrete([length, length])

        super().__init__()

    def sample_initial_state(self):
        return self.np_random.random(size=self.length)

    def reward_fn(self, state, act, next_state):
        num_correct = self._num_correct_positions(state)
        next_num_correct = self._num_correct_positions(next_state)
        potential_diff = next_num_correct - num_correct

        return int(self._is_sorted(next_state)) + potential_diff

    def transition_fn(self, state, action):
        next_state = state.copy()
        i, j = action
        next_state[[i, j]] = next_state[[j, i]]
        return next_state

    def _is_sorted(self, arr):
        return list(arr) == sorted(arr)

    def _num_correct_positions(self, arr):
        return np.sum(arr == sorted(arr))


_horizon_v0 = 6

gym.register(
    id=f"seals/Sort-v0",
    entry_point=f"derail.envs:SortEnv",
    max_episode_steps=_horizon_v0,
)
