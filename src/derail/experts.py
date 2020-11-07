import numpy as np

from derail.utils import (
    LightweightRLModel,
    force_shape,
    get_horizon,
    get_random_policy,
    get_raw_env,
    get_reward_matrix,
    get_transition_matrix,
)

from scipy.special import logsumexp

def get_noisyobs_expert(venv, **kwargs):
    env = get_raw_env(venv)

    def predict_fn(ob, state=None, deterministic=False):
        pos = ob[:2]
        dx, dy = env._goal - pos

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


def get_proc_goal_expert(venv, **kwargs):
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

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_largest_sum_expert(venv, **kwargs):
    def predict_fn(ob, state=None, deterministic=False):
        n = len(ob)
        action = int(np.sum(ob[: n // 2]) > np.sum(ob[n // 2 :]))
        return action, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_selectionsort_expert(env=None, **kwargs):
    def predict_fn(ob, state=None, deterministic=False):
        if state is None:
            state = 0
        next_to_sort = state

        act = None
        while act is None and next_to_sort < len(ob):
            pos = next_to_sort + ob[next_to_sort:].argmin()
            if pos != next_to_sort:
                act = (pos, next_to_sort)
            next_to_sort += 1

        if act is None:
            act = (0, 0)

        act = np.array(act)
        return act, next_to_sort

    return LightweightRLModel(predict_fn=predict_fn, env=env)


def get_insertionsort_expert(env=None, **kwargs):
    def predict_fn(ob, state=None, deterministic=False):
        act = None
        for i in range(len(ob) - 1):
            if ob[i] > ob[i + 1]:
                act = (i, i + 1)
                break

        if act is None:
            act = (0, 0)

        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=env)


def get_early_term_pos_expert(venv, horizon=10, **kwargs):
    def predict_fn(ob, state=None, deterministic=False):
        if state is None:
            state = 0
        t = state

        act = int(horizon - t <= 2)

        state = t + 1
        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_early_term_neg_expert(venv, **kwargs):
    def predict_fn(ob, state=None, deterministic=False):
        act = 1
        return act, state

    return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_parabola_expert(venv, **kwargs):
    env = get_raw_env(venv)

    def get_target(ob):
        x, y, a, b, c = ob
        x += env._x_step
        target = a * x ** 2 + b * x + c
        return target

    def predict_fn(ob, state=None, deterministic=False):
        y = ob[1]
        act = get_target(ob) - y
        act = np.array([act])
        return act, state

    expert = LightweightRLModel(predict_fn=predict_fn, env=venv)
    return expert


def hard_value_iteration(venv, discount=1.0):
    horizon = get_horizon(venv)
    nS = venv.observation_space.n
    nA = venv.action_space.n

    reward_matrix = force_shape(get_reward_matrix(venv), (nS, nA, nS))
    dynamics = get_transition_matrix(venv)

    Q = np.empty((horizon, nS, nA))
    V = np.empty((horizon + 1, nS))
    V[-1] = np.zeros(nS)
    for t in reversed(range(horizon)):
        for s in range(nS):
            for a in range(nA):
                Q[t, s, a] = dynamics[s, a, :] @ (reward_matrix[s, a, :] + discount * V[t + 1, :])
        V[t] = np.max(Q[t], axis=1)

    policy = np.eye(nA)[Q.argmax(axis=2)]

    return policy


def soft_value_iteration(venv, beta=10):
    horizon = get_horizon(venv)
    nS = venv.observation_space.n
    nA = venv.action_space.n

    reward_matrix = force_shape(get_reward_matrix(venv), (nS, nA, nS))
    dynamics = get_transition_matrix(venv)

    Q = np.empty((horizon, nS, nA))
    V = np.empty((horizon + 1, nS))
    V[-1] = np.zeros(nS)
    for t in reversed(range(horizon)):
        for s in range(nS):
            for a in range(nA):
                Q[t, s, a] = dynamics[s, a, :] @ (reward_matrix[s, a, :] + V[t + 1, :])
        V[t] = logsumexp(Q[t], axis=1)

    policy = np.exp(Q - V[:-1, :, None])

    return policy

def hard_mdp_expert(venv, *args, **kwargs):
    return LightweightRLModel.from_matrix(hard_value_iteration(venv), env=venv)

def soft_mdp_expert(venv, *args, **kwargs):
    return LightweightRLModel.from_matrix(soft_value_iteration(venv), env=venv)

