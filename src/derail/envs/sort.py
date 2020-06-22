import gym
from gym.spaces import MultiDiscrete, Box
from gym.utils import seeding
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


def get_selectionsort_expert(env=None):
    def predict_fn(ob, state=None, deterministic=False):
        if state is None:
            state = 0
        next_to_sort = state

        act = None
        while act is None and next_to_sort < len(ob):
            pos = next_to_sort + ob[next_to_sort:].argmin()
            if pos != next_to_sort:
                act = (pos, next_to_sort)
            next_to_sort += 1

        if act is None:
            act = (0, 0)

        act = np.array(act)
        return act, next_to_sort

    return LightweightRLModel(predict_fn=predict_fn, env=env)


def get_insertionsort_expert(env=None):
    def predict_fn(ob, state=None, deterministic=False):
        act = None
        for i in range(len(ob) - 1):
            if ob[i] > ob[i + 1]:
                act = (i, i + 1)
                break

        if act is None:
            act = (0, 0)

        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=env)


_horizon_v0 = 6

gym.register(
    id=f"seals/Sort-v0",
    entry_point=f"derail.envs:SortEnv",
    max_episode_steps=_horizon_v0,
)
