import numpy as np

from derail.utils import get_raw_env, LightweightRLModel

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

    def cross_entropy(ob, act):
        expert_act, _ = predict_fn(ob)
        stddev = env._x_step
        return ((act - expert_act)**2) / (2 * stddev**2)

    def predict_fn(ob, state=None, deterministic=False):
        y = ob[1]
        act = get_target(ob) - y
        act = np.array([act])
        return act, state

    expert = LightweightRLModel(predict_fn=predict_fn, env=venv)
    setattr(expert, 'cross_entropy', cross_entropy)
    return expert
