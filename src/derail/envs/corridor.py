
from gym import spaces

from derail.utils import *
from derail.envs import base_envs

class CorridorEnv(base_envs.ResettableMDP):
    def __init__(self, length=100, reward_period=None):
        if reward_period is None:
            reward_period = length + 1

        self._length = length
        self._reward_period = reward_period

        super().__init__(
            state_space=spaces.Discrete(self._length + 1),
            action_space=spaces.Discrete(2),
        )

    def terminal(self, state: int, n_actions_taken: int) -> bool:
        """Always returns False."""
        return False

    def initial_state(self) -> np.ndarray:
        return 0

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        return (state // self._reward_period) * (self._reward_period / (self._length + 1))

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Update x according to x_step and y according to action."""
        dx = (-1, 1)[action]
        return np.clip(state + dx, 0, self._length)
