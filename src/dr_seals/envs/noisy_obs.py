import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import numpy as np

from dr_seals.envs.base_env import BaseEnv

from dr_seals.utils import (
    get_raw_env,
    grid_transition_fn,
    LightweightRLModel,
)


class NoisyObsEnv(BaseEnv):
    def __init__(self, *args, size=5, noise_length=50, **kwargs):
        self.size = size
        self.noise_length = noise_length
        self.goal = np.array([self.size - 2, self.size - 2])

        self.observation_space = Box(
            low=np.concatenate(([0, 0], np.full(self.noise_length, -np.inf),)),
            high=np.concatenate(
                ([size - 1, size - 1], np.full(self.noise_length, np.inf),)
            ),
            dtype=float,
        )
        super().__init__(num_actions=5)

    def sample_initial_state(self):
        return np.array((0, 0))

    def reward_fn(self, state, act, next_state):
        dist = np.linalg.norm(self.goal - state)
        reward = int(dist < 1e-5)
        return reward

    def transition_fn(self, state, action):
        return grid_transition_fn(
            state, action, x_bounds=(0, self.size - 1), y_bounds=(0, self.size - 1)
        )

    def ob_from_state(self, state):
        noise_vector = self.np_random.randn(self.noise_length)
        ob = np.concatenate([state, noise_vector])
        return ob

    def state_from_ob(self, ob):
        return ob[:2]


def get_noisyobs_expert(venv):
    env = get_raw_env(venv)

    def predict_fn(ob, state=None, deterministic=False):
        pos = ob[:2]
        dx, dy = env.goal - pos

        conditions = [
            dx > 0,
            dy > 0,
            dx < 0,
            dy < 0,
            True,
        ]
        act = np.argmax(conditions)

        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


_horizon_v0 = 15

gym.register(
    id=f"seals/NoisyObs-v0",
    entry_point=f"dr_seals.envs:NoisyObsEnv",
    max_episode_steps=_horizon_v0,
)
