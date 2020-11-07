import time

from gym.spaces import Discrete

import numpy as np

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

from imitation.util.rollout import make_sample_until, generate_trajectories


## Getters

def get_internal_env(env):
    if hasattr(env, "venv"): return env.venv
    elif hasattr(env, "envs"): return env.envs[0]
    elif hasattr(env, "env"): return env.env
    else: return env

def fixpoint(f, x):
    nx = f(x)
    return x if nx == x else fixpoint(f, nx)

def get_raw_env(env):
    return fixpoint(get_internal_env, env)

def get_horizon(env):
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    if hasattr(env, "horizon"):
        return env.horizon
    elif get_internal_env(env) != env:
        return get_horizon(get_internal_env(env))
    else:
        # Estimate horizon
        trajs = sample_trajectories(env, get_random_policy(env))
        horizon = sum(len(traj.obs) - 1 for traj in trajs) // len(trajs)
        return horizon

def get_reward_matrix(env):
    if hasattr(env, "reward_matrix"):
        return env.reward_matrix
    elif get_internal_env(env) != env:
        return get_reward_matrix(get_internal_env(env))
    else:
        # Compute matrix from env.reward_fn
        nS = env.observation_space.n
        nA = env.action_space.n
        reward_matrix = np.empty((nS, nA, nS))
        S = range(nS)
        A = range(nA)

        for s, a, sn in itertools.product(S, A, S):
            reward_matrix[s, a, sn] = env.reward_fn(s, a, sn)
        return reward_matrix


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


## Evaluation, sampling

def monte_carlo_eval_policy(policy, env, **kwargs):
    rew, _ = evaluate_policy(policy, env, **kwargs)
    return rew

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


## Policies

class LightweightRLModel:
    def __init__(self, predict_fn, env=None, undo_vec=True):
        self.predict_fn = predict_fn
        self.env = env
        self._undo_vec = undo_vec

    def predict(self, ob, state=None, *args, **kwargs):
        ob = np.array(ob)
        undo_vec = self._undo_vec and len(ob.shape) > len(self.env.observation_space.shape)
        if undo_vec:
            ob = ob[0]
            if state is not None:
                state = state[0]

        action, state = self.predict_fn(ob, state, *args, **kwargs)
        if undo_vec:
            return [action], [state]
        else:
            return action, state

    def __call__(self, ob):
        act, _ = self.predict(ob)
        return act

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
                return model.predict(ob, state, *args, **kwargs)
        return LightweightRLModel(predict_fn=predict_fn, env=venv, undo_vec=False)


def get_ppo(env, *args, policy=MlpPolicy, learning_rate=1e-3, **kwargs):
    return PPO2(policy, env, *args, **kwargs)

def train_rl(env, *args, policy_fn=get_ppo, total_timesteps=10000, n_envs=1, **kwargs):
    model = policy_fn(env, *args, **kwargs)
    model.learn(total_timesteps)
    return model


## Misc

def sample_distribution(p):
    """Samples an integer with probabilities given by p."""
    return np.random.choice(np.arange(len(p)), p=p)

def force_shape(arr, shape):
    if arr.shape == shape:
        return arr

    new_arr = np.empty(shape)

    if len(arr.shape) == 1:
        new_arr[:] = arr[:, None, None]
    elif len(arr.shape) == 2:
        new_arr[:] = arr[:, :, None]

    return new_arr
