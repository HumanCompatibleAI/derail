"""Environment testing for correct behavior under stochasticity."""

import gym
import numpy as np

from derail.envs.base_env import BaseEnv


class RiskyPathEnv(BaseEnv):
    """

    Many LfH algorithms are derived from Maximum Entropy Inverse Reinforcement
    Learning (Ziebart et. al, 2008), which models the demonstrator as producing
    trajectories with probability p(tau) proportional to Exp(R(tau)).  This
    model implies that a demonstrator can ``control'' the environment well
    enough to follow any high-reward trajectory with high probability (Ziebart,
    2010). However, in stochastic environments, the agent cannot control the
    probability of each trajectory independently.  This misspecification may
    lead to poor behavior.

    This task tests for this behavior. The agent starts at s_0 and can reach
    the goal s_2 (reward 1.0) by either taking the safe path s_0 to s_1 to s_2,
    or taking a risky action, which has equal chances of going to either s_3
    (reward -100.0) or s_2.  The safe path has the highest expected return, but
    the risky action sometimes reaches the goal s_2 in fewer timesteps, leading
    to higher best-case return.  Algorithms that fail to correctly handle
    stochastic dynamics may therefore wrongly believe the reward favors taking
    the risky path. 
    """

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
    entry_point=f"derail.envs:RiskyPathEnv",
    max_episode_steps=_horizon_v0,
)
