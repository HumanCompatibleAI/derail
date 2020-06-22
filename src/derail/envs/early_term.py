"""Environment checking for correctness under early termination."""

from functools import partial

import gym
import numpy as np

from derail.envs.base_env import BaseEnv

from derail.utils import (
    LightweightRLModel,
    sample_distribution,
)


class EarlyTerminationEnv(BaseEnv):
    """

    Many implementations of imitation learning algorithms incorrectly assign a
    value of zero to terminal states (Kostrikov et. al (2018)).  Depending on
    the sign of the learned reward function in non-terminal states, this can
    either bias the agent to end episodes early or prolong them as long as
    possible.  This confounds evaluation as performance is spuriously high in
    tasks where the termination bias aligns with the task objective.  These
    tasks attempt to detect this type of bias, and they are adapted from
    Kostrikov et. al (2018).

    The environment is a 3-state MDP, in which the agent can either alternate
    between two initial states until reaching the time horizon, or they can
    move to a terminal state causing the episode to terminate early.
    """

    def __init__(self, is_reward_positive=True):
        nS = 3
        nA = 2

        self.transition_matrix = np.zeros((nS, nA, nS))

        self.transition_matrix[0, :, 1] = 1.0

        self.transition_matrix[1, 0, 0] = 1.0
        self.transition_matrix[1, 1, 2] = 1.0

        self.transition_matrix[2, :, 2] = 1.0

        if is_reward_positive:
            rewards = np.array([1.0, 1.0, 1.0])
        else:
            rewards = np.array([-1.0, -1.0, -1.0])

        self.reward_matrix = np.empty((nS, nA, nS))
        self.reward_matrix[:, :] = rewards

        super().__init__(num_states=nS, num_actions=nA)

    def termination_fn(self, state):
        return self.state == self.observation_space.n - 1


def get_early_term_pos_expert(venv, horizon=10):
    def predict_fn(ob, state=None, deterministic=False):
        if state is None:
            state = 0
        t = state

        act = int(horizon - t <= 2)

        state = t + 1
        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_early_term_neg_expert(venv):
    def predict_fn(ob, state=None, deterministic=False):
        act = 1
        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


EarlyTermPosEnv = partial(EarlyTerminationEnv, is_reward_positive=True)
EarlyTermNegEnv = partial(EarlyTerminationEnv, is_reward_positive=False)

_horizon_v0 = 10

gym.register(
    id=f"seals/EarlyTermPos-v0",
    entry_point=f"derail.envs:EarlyTermPosEnv",
    max_episode_steps=_horizon_v0,
)

gym.register(
    id=f"seals/EarlyTermNeg-v0",
    entry_point=f"derail.envs:EarlyTermNegEnv",
    max_episode_steps=_horizon_v0,
)
