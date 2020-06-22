import gym
from gym.spaces import Discrete
from gym.utils import seeding
import numpy as np

from derail.utils import sample_distribution


class BaseEnv(gym.Env):
    def __init__(self, num_states=None, num_actions=None):
        super().__init__()

        if num_states is not None:
            self.observation_space = Discrete(num_states)
        if num_actions is not None:
            self.action_space = Discrete(num_actions)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.sample_initial_state()
        return self.ob_from_state(self.state)

    def step(self, act):
        assert act in self.action_space, f"{act} not in {self.action_space}"

        old_state = self.state
        self.state = self.transition_fn(self.state, act)
        next_ob = self.ob_from_state(self.state)

        reward = self.reward_fn(old_state, act, self.state)

        done = self.termination_fn(self.state)
        info = {}

        return next_ob, reward, done, info

    def reward_fn(self, state, action, new_state):
        return self.reward_matrix[state, action, new_state]

    def transition_fn(self, state, action):
        return sample_distribution(
            self.transition_matrix[state, action], random=self.np_random
        )

    def ob_from_state(self, state):
        return state

    def state_from_ob(self, ob):
        return ob

    def sample_initial_state(self):
        return np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )

    def initial_state_distribution(self):
        initial_state = self.sample_initial_state()
        nS = self.observation_space.n
        one_hot_state = np.eye(nS)[initial_state]
        return one_hot_state

    def termination_fn(self, state):
        return False

    def render(self):
        print(self.state)
