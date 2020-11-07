import functools
import numpy as np

from scipy.special import logsumexp

from derail.utils import (
    get_horizon,
    get_initial_state_dist,
    get_transition_matrix,
    LightweightRLModel,
    sample_trajectories,
)


class LinearRewardModel:
    def __init__(self, state_features):
        self.state_features = state_features
        nF, nS = state_features.shape
        self._w = np.random.randn(nF)

    def get_state_rewards(self):
        return self._w @ self.state_features

    def get_state_reward_grads(self):
        return self.state_features

    def get_rewards_and_grads(self):
        return (self.get_state_rewards(), self.get_state_reward_grads())

    def update_params(self, alpha, grad):
        self._w += alpha * grad.reshape(self._w.shape)

    def reward_fn(self, ob, act, next_ob):
        return self._w[next_ob]



def maximum_entropy_irl(
    venv,
    expert=None,
    expert_venv=None,
    expert_trajectories=None,
    causal=True,
    total_timesteps=10000,
    **kwargs,
):
    if expert_trajectories is None:
        expert_trajectories = sample_trajectories(
            expert_venv, expert, n_timesteps=total_timesteps
        )

    nS = venv.observation_space.n

    expert_occupancy = np.zeros(nS)
    for trj in expert_trajectories:
        for ob in trj.obs:
            expert_occupancy[ob] += 1.0
    expert_occupancy /= expert_occupancy.sum()

    state_features = np.identity(nS)
    reward_model = LinearRewardModel(state_features)

    q_update_fn = mce_q_update_fn if causal else max_ent_q_update_fn

    horizon = get_horizon(venv)
    initial_state_distribution = get_initial_state_dist(venv)

    irl_reward, policy_matrix = occupancy_match_irl(
        dynamics=get_transition_matrix(venv),
        horizon=horizon,
        reward_model=reward_model,
        expert_occupancy=expert_occupancy,
        initial_state_distribution=initial_state_distribution,
        max_iterations=total_timesteps,
        q_update_fn=q_update_fn,
    )

    policy = LightweightRLModel.from_matrix(policy_matrix, env=venv)

    results = {}
    results["reward_model"] = irl_reward
    results["policy"] = policy
    return results


def compute_occupancy_measure(
    transition, policy, state_rewards, initial_state_distribution
):
    horizon = policy.shape[0]
    nS = transition.shape[0]
    # transport[t, s, ns] == sum_a policy[t, s, a] * transition[s, a, ns]
    # transport = np.sum(policy[:, :, :, None] * transition[None, :, :, :], axis=2)
    transport = np.einsum("tsa,san->tsn", policy, transition)

    density = np.zeros((horizon + 1, nS))
    density[0] = initial_state_distribution
    for t in range(horizon):
        density[t + 1, :] = density[t, :] @ transport[t, :, :]

    return density.sum(axis=0) / density.sum()


def occupancy_match_irl(
    dynamics,
    horizon,
    reward_model,
    expert_occupancy,
    q_update_fn,
    initial_state_distribution,
    max_iterations=10000,
):
    """Maximum Entropy Inverse Reinforcement Learning.
    """
    nS, nA, _ = dynamics.shape
    nF = len(reward_model._w)

    def compute_policy(state_rewards):
        Q = np.empty((horizon, nS, nA))
        V = np.empty((horizon + 1, nS))

        V[-1] = logsumexp(state_rewards[:, None], axis=1)

        for t in reversed(range(horizon)):
            Q[t] = q_update_fn(V[t + 1], state_rewards, dynamics)
            V[t] = logsumexp(Q[t], axis=1)

        policy = np.exp(Q - V[:-1, :, None])

        return policy

    EPS = 1e-5
    occupancy_diff_max = np.inf
    grad_norm = np.inf

    # Adam params
    alpha = 1e-3
    beta_1 = 0.9
    beta_2 = 0.99
    m = np.zeros(nF)
    v = np.zeros(nF)

    iter_step = 0
    while occupancy_diff_max > EPS and grad_norm > EPS and iter_step < max_iterations:
        state_rewards, state_reward_grads = reward_model.get_rewards_and_grads()
        policy = compute_policy(state_rewards)
        learner_occupancy = compute_occupancy_measure(
            dynamics, policy, state_rewards, initial_state_distribution
        )

        grad = state_reward_grads @ (expert_occupancy - learner_occupancy)

        # Adam update
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad ** 2
        m_hat = m / (1 - beta_1)
        v_hat = v / (1 - beta_2)
        reward_model._w += alpha * m_hat / (np.sqrt(v_hat) + 1e-6)

        occupancy_diff_max = np.max(np.abs(expert_occupancy - learner_occupancy))
        grad_norm = np.linalg.norm(grad)

        iter_step += 1

    policy = compute_policy(state_rewards)
    return reward_model, policy


def max_ent_q_update_fn(values, state_rewards, dynamics):
    return state_rewards[:, None] + logsumexp(
        np.log(dynamics + 1e-50) + values[None, None, :], axis=2
    )


def mce_q_update_fn(values, state_rewards, dynamics):
    return state_rewards[:, None] + dynamics @ values


max_ent_irl = functools.partial(maximum_entropy_irl, causal=False)
mce_irl = functools.partial(maximum_entropy_irl, causal=True)
