from datetime import datetime
import functools
import operator
import time

from gym.spaces import Discrete

import numpy as np
from scipy.special import logsumexp

import tensorflow as tf

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

from imitation.util.rollout import make_sample_until, generate_trajectories

TIMESTAMP = None

def get_last_timestamp():
    return TIMESTAMP

def get_timestamp():
    global TIMESTAMP
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    return TIMESTAMP


def sample_distribution(p, random=None):
    """Samples an integer with probabilities given by p."""
    if random is None:
        random = np.random
    return random.choice(np.arange(len(p)), p=p)


def monte_carlo_eval_policy(policy, env, **kwargs):
    rew, _ = evaluate_policy(policy, env, **kwargs)
    return rew


class RunningMeanVar:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

        self.mean = 0
        self.var = 1.0
        self.count = 0

        self.update_on = True

    def count_update(self, xs):
        batch_mean = np.mean(xs)
        batch_var = np.var(xs)
        batch_count = len(xs)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def exp_update(self, xs):
        if self.update_on:
            batch_mean = np.mean(xs)
            batch_var = np.var(xs)

            delta = batch_mean - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta**2)

        return (xs - self.mean) / np.sqrt(self.var)


def force_shape(arr, shape):
    if arr.shape == shape:
        return arr

    new_arr = np.empty(shape)

    if len(arr.shape) == 1:
        new_arr[:] = arr[:, None, None]
    elif len(arr.shape) == 2:
        new_arr[:] = arr[:, :, None]

    return new_arr

def make_egreedy(model, venv, epsilon=0.1):
    if hasattr(model, 'policy_matrix'):
        nA = model.policy_matrix.shape[-1]
        new_policy = (1 - epsilon) * model.policy_matrix + (epsilon / nA) * np.ones_like(model.policy_matrix)
        return LightweightRLModel.from_matrix(new_policy, env=venv)
    else:
        random_policy = get_random_policy(venv)
        def predict_fn(ob, state, *args, **kwargs):
            if np.random.rand() < epsilon:
                return random_policy.predict(ob, state, *args, **kwargs)
            else:
                return model.predict(np.array([ob]), np.array([state]), *args, **kwargs)
        return LightweightRLModel(predict_fn=predict_fn, env=venv, undo_vec=False)

def get_raw_policy(policy):
    if hasattr(policy, "policy_matrix"):
        return policy.policy_matrix
    elif hasattr(policy, "action_probability"):
        env = policy.env
        states = list(range(env.observation_space.n))
        probs = policy.action_probability(states)
        matrix = np.empty((get_horizon(env), *probs.shape))
        matrix[:] = probs
        return matrix
    else:
        return policy

def get_initial_state_dist(env):
    return get_raw_env(env).initial_state_dist

def get_transition_matrix(env):
    return get_raw_env(env).transition_matrix

def tabular_eval_policy(policy, env, **kwargs):
    policy = get_raw_policy(policy)

    if not isinstance(policy, np.ndarray):
        return monte_carlo_eval_policy(policy, env, **kwargs)

    occupancy = get_initial_state_dist(env)

    returns = 0

    rewards = get_reward_matrix(env)

    transition = get_transition_matrix(env)
    rewards = force_shape(rewards, transition.shape)

    horizon = get_horizon(env)
    state_action_rewards = np.sum(transition * rewards, axis=2)
    transport = np.sum(policy[:, :, :, None] * transition[None, :, :, :], axis=2)

    for t in range(horizon):
        returns += occupancy @ policy[t] @ (occupancy @ state_action_rewards)
        occupancy = occupancy @ transport[t]

    return returns


# todo: remove code duplication
def ti_hard_value_fn(venv, discount=0.9, num_iter=200):
    """Time-independent value function"""

    env = get_raw_env(venv)

    horizon = get_horizon(venv)
    nS = env.observation_space.n
    nA = env.action_space.n

    reward_matrix = get_reward_matrix(env)
    reward_matrix = force_shape(reward_matrix, (nS, nA, nS))

    dynamics = env.transition_matrix

    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        for s in range(nS):
            for a in range(nA):
                Q[s, a] = dynamics[s, a, :] @ (reward_matrix[s, a, :] + discount * V[:])
        V = np.max(Q, axis=1)

    return V

def ti_soft_value_fn(venv, discount=0.9, beta=10, num_iter=200):
    """Time-independent value function"""

    env = get_raw_env(venv)

    horizon = get_horizon(venv)
    nS = env.observation_space.n
    nA = env.action_space.n

    R = get_reward_matrix(env)
    R = force_shape(R, (nS, nA, nS))

    T = env.transition_matrix

    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = np.sum(T * R, axis=2) + discount * np.tensordot(T, V, axes=(2, 0))
        V = logsumexp(beta * Q, axis=1) / beta

    policy = np.exp(beta * (Q - V[:, None]))
    policy /= policy.sum(axis=1, keepdims=True)

    return policy, {'V' : V, 'Q' : Q}

def hard_value_iteration(venv, discount=1.0):
    env = get_raw_env(venv)

    horizon = get_horizon(venv)
    nS = env.observation_space.n
    nA = env.action_space.n

    reward_matrix = get_reward_matrix(env)
    reward_matrix = force_shape(reward_matrix, (nS, nA, nS))

    dynamics = env.transition_matrix

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
    env = get_raw_env(venv)

    horizon = get_horizon(venv)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    reward_matrix = get_reward_matrix(env)
    dynamics = env.transition_matrix

    Q = np.empty((horizon, num_states, num_actions))
    V = np.empty((horizon + 1, num_states))

    V[-1] = np.zeros(num_states)

    for t in reversed(range(horizon)):
        for s in range(num_states):
            for a in range(num_actions):
                Q[t, s, a] = dynamics[s, a, :] @ (reward_matrix[s, a, :] + V[t + 1, :])
        V[t] = logsumexp(Q[t], axis=1)

    policy = np.exp(Q - V[:-1, :, None])

    return policy


def get_internal_env(env):
    if hasattr(env, "venv"): return env.venv
    elif hasattr(env, "envs"): return env.envs[0]
    elif hasattr(env, "env"): return env.env
    else: return env

def get_raw_env(env):
    internal_env = get_internal_env(env)
    return env if internal_env == env else get_raw_env(internal_env)

def get_horizon(env):
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    if hasattr(env, "horizon"):
        return env.horizon
    elif get_internal_env(env) != env:
        return get_horizon(get_internal_env(env))
    else:
        trajs = sample_trajectories(env, get_random_policy(env))
        horizon = sum(len(traj.obs) - 1 for traj in trajs) // len(trajs)
        return horizon

def get_reward_matrix(env):
    if hasattr(env, "reward_matrix"):
        return env.reward_matrix
    if hasattr(env, "get_reward_matrix"):
        return env.get_reward_matrix()
    elif get_internal_env(env) != env:
        return get_reward_matrix(get_internal_env(env))
    else:
        nS = env.observation_space.n
        nA = env.action_space.n
        reward_matrix = np.empty((nS, nA, nS))
        S = range(nS)
        A = range(nA)

        for s, a, sn in itertools.product(S, A, S):
            reward_matrix[s, a, sn] = env.reward_fn(s, a, sn)
        return reward_matrix



def get_mdp_expert(venv, is_hard=True, **kwargs):
    policy = hard_value_iteration(venv) if is_hard else soft_value_iteration(venv)
    return LightweightRLModel.from_matrix(policy, env=venv)


get_hard_mdp_expert = functools.partial(get_mdp_expert, is_hard=True)
get_soft_mdp_expert = functools.partial(get_mdp_expert, is_hard=False)


class LinearRewardModel:
    def __init__(self, state_features):
        self.state_features = state_features
        num_features, num_states = state_features.shape
        self._w = np.random.randn(num_features)

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


def sample_trajectories(env, expert, n_episodes=None, n_timesteps=None):
    if n_episodes is None and n_timesteps is None:
        n_episodes = 20

    expert_trajectories = generate_trajectories(
        expert,
        env,
        sample_until=make_sample_until(n_episodes=n_episodes, n_timesteps=n_timesteps),
    )
    return expert_trajectories



def render_trajectories(env, policy, n_episodes=5, dt=0.0):
    for i in range(n_episodes):
        print(f"----- Episode {i} -----")

        ob = env.reset()
        env.render()
        time.sleep(dt)

        state = None
        done = False
        while not done:
            ac, state = policy.predict(ob, state)
            ob, re, done, _ = env.step(ac)
            env.render()
            time.sleep(dt)

    return None

def prod(seq):
    return functools.reduce(operator.mul, seq, 1)

def get_num_actions(env):
    if hasattr(env.action_space, 'n'):
        return env.action_space.n
    else:
        return prod(env.action_space.nvec)

class LightweightRLModel:
    def __init__(self, predict_fn, env=None, undo_vec=True):
        self.predict_fn = predict_fn
        self.env = env
        self._undo_vec = undo_vec

    def predict(self, ob, state=None, *args, **kwargs):
        # if self.is_vec:
        ob = np.array(ob)
        undo_vec  = self._undo_vec and len(ob.shape) > len(self.env.observation_space.shape)
        if undo_vec:
            ob = ob[0]
            if state is not None:
                state = state[0]

        action, state = self.predict_fn(ob, state, *args, **kwargs)
        if undo_vec:
            return [action], [state]
        else:
            return action, state

    def cross_entropy(self, ob, act):
        nA = get_num_actions(self.env)
        EPS = 1e-1
        expert_act = self.predict(ob)
        if np.all(act == expert_act):
            return np.log(1 - EPS + EPS / nA)
        else:
            return np.log(EPS / nA)

    def __call__(self, ob):
        act, _ = self.predict(ob)
        return act

    @property
    def ob_space(self):
        return self.env.observation_space

    @property
    def ac_space(self):
        return self.env.action_space

    @classmethod
    def from_matrix(cls, policy_matrix, env=None):
        def predict_fn(ob, state=None, deterministic=False):
            if len(policy_matrix.shape) == 3:
                t = state if state is not None else 0
                action_distribution = policy_matrix[t, ob]
                new_state = t + 1
            else:
                action_distribution = policy_matrix[ob]
                new_state = state

            if deterministic:
                act = np.argmax(action_distribution)
            else:
                act = sample_distribution(action_distribution)
            return act, new_state

        model = cls(predict_fn=predict_fn, env=env)
        model.policy_matrix = policy_matrix
        return model


def get_random_policy(venv, tabular=True):
    env = get_raw_env(venv)

    tabular = (
        tabular
        and isinstance(env.observation_space, Discrete)
        and isinstance(env.action_space, Discrete)
    )

    if tabular:
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        matrix = np.full((get_horizon(venv), num_states, num_actions), 1 / num_actions)
        return LightweightRLModel.from_matrix(matrix, env=venv)
    else:

        def predict_fn(ob, state=None, deterministic=False):
            action = env.action_space.sample()
            return action, state

        return LightweightRLModel(predict_fn=predict_fn, env=venv)


def get_ppo(env, *args, policy=MlpPolicy, learning_rate=1e-3, **kwargs):
    from stable_baselines import PPO2

    return PPO2(policy, env, *args, **kwargs)


def train_rl(env, *args, policy_fn=get_ppo, total_timesteps=10000, n_envs=1, **kwargs):
    model = policy_fn(env, *args, **kwargs)
    model.learn(total_timesteps)
    return model


def ppo_algo(env, *args, policy_fn=get_ppo, total_timesteps=10000, **kwargs):
    policy = train_rl(env, policy_fn=policy_fn, total_timesteps=total_timesteps)
    return {"policy": policy, "reward_model": "groundtruth"}


def get_expert_algo(env, expert, *args, **kwargs):
    return {"policy": expert, "reward_model": "groundtruth"}


def grid_transition_fn(
    state, action, x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)
):
    dirs = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (0, 0),
    ]

    x, y = state
    dx, dy = dirs[action]

    next_x = np.clip(x + dx, *x_bounds)
    next_y = np.clip(y + dy, *y_bounds)
    next_state = np.array([next_x, next_y], dtype=state.dtype)

    return next_state


def one_hot_encoding(pos: int, size: int) -> np.ndarray:
    """Returns a 1-D hot encoding of a given position and size."""
    return np.eye(size)[pos]

