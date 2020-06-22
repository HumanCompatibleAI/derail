from functools import partial

import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import sample_distribution


class InitStateShiftEnv(BaseEnv):
    def __init__(self, initial_state=0):
        self.initial_state = initial_state
        nS = 7
        nA = 2

        rewards = np.array([0, 0, 0, 1, -1, -1, 2])
        self.reward_matrix = np.empty((nS, nA, nS))
        self.reward_matrix[:, :] = rewards

        T = np.zeros((nS, nA, nS))

        for state in (0, 1, 2):
            for action in range(nA):
                next_state = 2 * state + 1 + action
                T[state, action, next_state] = 1.0

        absorb = np.arange(3, 7)
        T[absorb, :, absorb] = 1.0

        self.transition_matrix = T

        super().__init__(num_states=nS, num_actions=nA)

    def sample_initial_state(self):
        return self.initial_state


InitStateShiftExpertEnv = partial(InitStateShiftEnv, initial_state=0)
InitStateShiftLearnerEnv = partial(InitStateShiftEnv, initial_state=1)

_horizon_v0 = 2

gym.register(
    id=f"seals/InitStateShiftLearner-v0",
    entry_point=f"derail.envs:InitStateShiftLearnerEnv",
    max_episode_steps=_horizon_v0,
)

gym.register(
    id=f"seals/InitStateShiftExpert-v0",
    entry_point=f"derail.envs:InitStateShiftExpertEnv",
    max_episode_steps=_horizon_v0,
)
