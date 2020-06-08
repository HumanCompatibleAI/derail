import gym
from gym.spaces import Discrete
from gym.utils import seeding
import numpy as np

from dr_seals.envs.base_env import BaseEnv


class RiskyPathEnv(BaseEnv):
    def __init__(self):
        nS = 4
        nA = 2

        self.transition_matrix = np.zeros((nS, nA, nS))
        self.transition_matrix[0, 0, 1] = 1.0
        self.transition_matrix[0, 1, [2, 3]] = 0.5

        self.transition_matrix[1, 0, 2] = 1.0
        self.transition_matrix[1, 1, 1] = 1.0

        self.transition_matrix[[2, 3], :, [2, 3]] = 1.0

        self.reward_matrix = np.zeros((nS, nA, nS))
        self.reward_matrix[2, :, :] = 1.0
        self.reward_matrix[3, :, :] = -100.0

        super().__init__(num_states=nS, num_actions=nA)


_horizon_v0 = 5

gym.register(
    id=f"seals/RiskyPath-v0",
    entry_point=f"dr_seals.envs:RiskyPathEnv",
    max_episode_steps=_horizon_v0,
)
