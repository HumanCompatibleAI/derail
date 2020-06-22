import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import LightweightRLModel


class EvenOddEnv(BaseEnv):
    def __init__(self, length=50):
        self.length = length
        self.observation_space = Box(low=0.0, high=1.0, shape=(length,))
        super().__init__(num_actions=2)

    def sample_initial_state(self):
        return self.np_random.rand(self.length)

    def reward_fn(self, state, act, next_state):
        label = np.sum(state[::2]) < np.sum(state[1::2])
        return int(act == label)

    def transition_fn(self, state, action):
        return state


def get_evenodd_expert(venv):
    def predict_fn(ob, state=None, deterministic=False):
        return int(np.sum(ob[::2]) < np.sum(ob[1::2])), state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


_horizon_v0 = 1

gym.register(
    id=f"seals/EvenOdd-v0",
    entry_point=f"derail.envs:EvenOddEnv",
    max_episode_steps=_horizon_v0,
)
