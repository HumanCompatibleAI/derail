import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from dr_seals.envs.base_env import BaseEnv

from dr_seals.utils import (
    grid_transition_fn,
    LightweightRLModel,
)


class RandomGoalEnv(BaseEnv):
    def __init__(self, bounds=100, distance=10):
        self.bounds = bounds
        self.distance = distance

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,))

        super().__init__(num_actions=5)

    def sample_initial_state(self):
        pos = self.np_random.randint(low=-self.bounds, high=self.bounds, size=(2,))

        x_dist = self.np_random.randint(self.distance)
        y_dist = self.distance - x_dist
        random_signs = lambda shape: 2 * self.np_random.randint(2, size=shape) - 1
        goal = pos + random_signs(2) * (x_dist, y_dist)

        return np.concatenate([pos, goal])

    def reward_fn(self, state, act, next_state):
        return (-1) * np.sum(np.abs(next_state[2:] - next_state[:2]))

    def transition_fn(self, state, action):
        pos, goal = state[:2], state[2:]
        next_pos = grid_transition_fn(pos, action)
        return np.concatenate([next_pos, goal])


def get_random_goal_expert(env):
    def predict_fn(ob, state=None, deterministic=False):
        pos, goal = ob[:2], ob[2:]
        dx, dy = goal - pos

        conditions = [
            dx > 0,
            dy > 0,
            dx < 0,
            dy < 0,
            True,
        ]
        act = np.argmax(conditions)

        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=env)


_horizon_v0 = 20

gym.register(
    id=f"seals/RandomGoal-v0",
    entry_point=f"dr_seals.envs:RandomGoalEnv",
    max_episode_steps=_horizon_v0,
)
