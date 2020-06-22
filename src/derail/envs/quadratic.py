import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import (
    get_raw_env,
    LightweightRLModel,
)


class QuadraticEnv(BaseEnv):
    def __init__(self, dx=0.05, bounds=5):
        self.dx = dx
        self.bounds = bounds

        self.observation_space = Box(low=-bounds, high=bounds, shape=(5,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))

        super().__init__()

    def sample_initial_state(self):
        a, b, c = -1 + 2 * self.np_random.rand(3)
        x, y = 0, c
        return np.array([x, y, a, b, c])

    def reward_fn(self, state, act, next_state):
        x, y, a, b, c = next_state
        target = a * x ** 2 + b * x + c
        # err = np.abs(y - target)
        err = (y - target) ** 2
        return (-1) * err

    def transition_fn(self, state, action):
        x, y, a, b, c = state
        next_x = x + self.dx
        next_y = np.clip(y + action, -self.bounds, self.bounds).squeeze()
        return np.array([next_x, next_y, a, b, c])


def get_quadratic_expert(venv):
    env = get_raw_env(venv)

    def predict_fn(ob, state=None, deterministic=False):
        x, y, a, b, c = ob
        x += env.dx
        target = a * x ** 2 + b * x + c
        act = target - y
        act = np.array([act])
        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


_horizon_v0 = 20

gym.register(
    id=f"seals/Quadratic-v0",
    entry_point=f"derail.envs:QuadraticEnv",
    max_episode_steps=_horizon_v0,
)
