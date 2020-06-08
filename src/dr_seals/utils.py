from datetime import datetime
import functools
import time

from gym.spaces import Discrete

import numpy as np
from scipy.special import logsumexp

import tensorflow as tf

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

from imitation.util.rollout import make_sample_until, generate_trajectories


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def sample_distribution(p, random=None):
    if random is None:
        random = np.random
    return random.choice(np.arange(len(p)), p=p)


def monte_carlo_eval_policy(policy, env, **kwargs):
    rew, _ = evaluate_policy(policy, env, **kwargs)
    return rew


def get_reward_matrix(env):
    if hasattr(env, "reward_matrix"):
        return env.reward_matrix
    if hasattr(env, "get_reward_matrix"):
        return env.get_reward_matrix()

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    reward_matrix = np.empty((num_states, num_actions, num_states))

    for state in range(num_states):
        for action in range(num_actions):
            for next_state in range(num_states):
                reward_matrix[state, action, next_state] = env.reward_fn(
                    state, action, next_state
                )

    return reward_matrix


def get_raw_policy(policy):
    if hasattr(policy, "policy_matrix"):
        return policy.policy_matrix
    elif hasattr(policy, "action_probability"):
        env = get_raw_env(policy.env)
        states = list(range(env.observation_space.n))
        probs = policy.action_probability(states)
        matrix = np.empty((get_horizon(env), *probs.shape))
        matrix[:] = probs
        return matrix
    else:
        return policy


def tabular_eval_policy(policy, env, **kwargs):
    env = get_raw_env(env)
    policy = get_raw_policy(policy)

    if not isinstance(policy, np.ndarray):
        return monte_carlo_eval_policy(policy, env, **kwargs)

    occupancy = env.initial_state_distribution()

    returns = 0

    rewards = get_reward_matrix(env)
    transition = env.transition_matrix

    horizon = get_horizon(env)
    state_action_rewards = np.sum(transition * rewards, axis=2)
    transport = np.sum(policy[:, :, :, None] * transition[None, :, :, :], axis=2)

    for t in range(horizon):
        returns += occupancy @ policy[t] @ (occupancy @ state_action_rewards)
        occupancy = occupancy @ transport[t]

    return returns


# todo: remove code duplication
def hard_value_iteration(env):
    horizon = get_horizon(env)
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
        V[t] = np.max(Q[t], axis=1)

    policy = np.eye(num_actions)[Q.argmax(axis=2)]

    return policy


def soft_value_iteration(env):
    horizon = get_horizon(env)
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


def get_raw_env(env):
    if hasattr(env, "venv"):
        return get_raw_env(env.venv)
    elif hasattr(env, "envs"):
        return env.envs[0]
    else:
        return env


def get_mdp_expert(venv, is_hard=True, **kwargs):
    env = get_raw_env(venv)

    policy = hard_value_iteration(env) if is_hard else soft_value_iteration(env)
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


def get_horizon(venv):
    env = get_raw_env(venv)
    if hasattr(env, "horizon"):
        return env.horizon
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    else:
        trajs = sample_trajectories(env, get_random_policy(env))
        horizon = sum(len(traj.obs) - 1 for traj in trajs) // len(trajs)
        return horizon


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


class LightweightRLModel:
    def __init__(self, predict_fn, env=None):
        self.predict_fn = predict_fn
        self.env = env

    def predict(self, ob, state=None, *args, **kwargs):
        # if self.is_vec:
        ob = np.array(ob)
        is_vec = len(ob.shape) > len(self.env.observation_space.shape)
        if is_vec:
            ob = ob[0]
            if state is not None:
                state = state[0]

        action, state = self.predict_fn(ob, state, *args, **kwargs)
        if is_vec:
            return [action], [state]
        else:
            return action, state

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
        matrix = np.full((get_horizon(env), num_states, num_actions), 1 / num_actions)
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
