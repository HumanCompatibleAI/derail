from collections import Counter, defaultdict, deque, namedtuple
import math
import functools

import numpy as np
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines import PPO2
from stable_baselines.common.base_class import ActorCriticRLModel
from stable_baselines.common.policies import MlpPolicy

from imitation.rewards.reward_net import BasicShapedRewardNet
from imitation.util import reward_wrapper, build_mlp, sequential

from derail.callbacks import Callback

from derail.utils import (
    get_random_policy,
    sample_trajectories,
    get_raw_env,
    get_horizon,
    ti_hard_value_fn,
    ti_soft_value_fn,
)

def preferences(
    venv,
    expert=None,
    evaluate_trajectories_fn=None,
    # n_pairs_per_batch=50,
    n_timesteps_per_query=500,
    reward_lr=1e-3,
    policy_lr=1e-3,
    # policy_epoch_timesteps=200,
    policy_epoch_timesteps=1000,
    total_timesteps=10000,
    cloning_bonus=False,
    state_only=False,
    callback=None,
    do_rnd=False,
    **kwargs,
):
    if callback is None:
        callback = Callback()

    if evaluate_trajectories_fn is None:
        reward_eval_fn = reward_eval_path_fn(venv)
        evaluate_trajectories_fn = get_eval_trajectories_fn(reward_eval_fn)

        if cloning_bonus:
            cloning_eval_fn = soft_expert_cloning_eval_fn(expert, venv)
            eval_fn = lambda path : reward_eval_fn(path) + cloning_eval_fn(path)
            evaluate_trajectories_fn = get_eval_trajectories_fn(eval_fn)

    # Create reward model
    rn = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        theta_units=[32, 32],
        phi_units=[32, 32],
        scale=True,
        state_only=state_only,
    )


    # Compute trajectory probabilities
    preferences_ph = tf.placeholder(
        shape=(None, 2), dtype=tf.float32, name="preferences",
    )
    num_segments = 2 * tf.shape(preferences_ph)[0]
    rewards_out = tf.reshape(rn.reward_output_train, [num_segments, -1])
    returns_out = tf.reduce_sum(rewards_out, axis=1)
    returns = tf.reshape(returns_out, shape=[-1, 2])
    log_probs = tf.nn.log_softmax(returns, axis=1)

    # Write loss and optimizer op
    loss = (-1) * tf.reduce_sum(log_probs * preferences_ph)
    optimizer = tf.train.AdamOptimizer(learning_rate=reward_lr)
    reward_train_op = optimizer.minimize(loss)


    reward_fn = get_reward_fn_from_model(rn)

    # Random network distillation bonus
    if do_rnd:
        rnd_target_net = build_mlp([32, 32, 32])
        rnd_target = sequential(rn.obs_inp, rnd_target_net)

        rnd_pred_net = build_mlp([32, 32, 32])
        rnd_pred = sequential(rn.obs_inp, rnd_pred_net)

        rnd_loss = tf.mean((tf.stop_gradient(rnd_target) - rnd_pred)**2)
        rnd_optimizer = tf.train.AdamOptimizer(learning_rate=reward_lr)
        rnd_train_op = rnd_optimizer.minimize(rnd_loss)

        def rnd_reward_fn(obs, *args, **kwargs):
            return sess.run(rnd_loss, feed_dict={rn.obs_ph : obs})

        extrinsic_reward_fn = reward_fn
        def reward_fn(*args, **kwargs):
            return extrinsic_reward_fn(*args, **kwargs) + rnd_reward_fn(*args, **kwargs)

    # Create learner from reward model
    venv_train = reward_wrapper.RewardVecEnvWrapper(venv, reward_fn)
    policy = PPO2(MlpPolicy, venv_train, learning_rate=policy_lr)


    # Start training
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    horizon = get_horizon(venv)

    num_epochs = int(np.ceil(total_timesteps / policy_epoch_timesteps))

    callback.start(locals(), globals())
    for epoch in range(num_epochs):
        callback.step(locals(), globals())

        n_pairs_per_batch = (n_timesteps_per_query / (2 * horizon))
        trajectories = sample_trajectories(venv, policy, 2 * n_pairs_per_batch)

        segments = get_segments(trajectories)

        seg_returns = evaluate_trajectories_fn(segments)
        seg_returns = seg_returns.reshape(-1, 2)
        preferences = np.stack(
            [
                seg_returns[:, 0] > seg_returns[:, 1],
                seg_returns[:, 1] > seg_returns[:, 0],
            ],
            axis=1,
        )

        obs = np.concatenate([seg.obs for seg in segments])
        acts = np.concatenate([seg.acts for seg in segments])
        next_obs = np.concatenate([seg.next_obs for seg in segments])

        sess.run(
            reward_train_op,
            feed_dict={
                rn.obs_ph: obs,
                rn.act_ph: acts,
                rn.next_obs_ph: next_obs,
                preferences_ph: preferences,
            },
        )

        # policy.set_env(venv_train)  # Possibly redundant?
        policy.learn(total_timesteps=policy_epoch_timesteps)
    epoch += 1

    callback.step(locals(), globals())

    results = {}
    results["reward_model"] = rn
    results["policy"] = policy

    callback.end(locals(), globals())

    return results


Segment = namedtuple("Segment", ["obs", "acts", "next_obs"])


def get_segments(trajectories):
    lengths = [len(trj.acts) for trj in trajectories]

    seg_len = functools.reduce(math.gcd, lengths)

    obs = np.concatenate([trj.obs[:-1] for trj in trajectories])
    acts = np.concatenate([trj.acts for trj in trajectories])
    next_obs = np.concatenate([trj.obs[1:] for trj in trajectories])

    num_segments = sum(lengths) // seg_len
    ob_shape = trajectories[0].obs[0].shape
    act_shape = trajectories[0].acts[0].shape
    new_obs_shape = (num_segments, seg_len, *ob_shape)
    new_acts_shape = (num_segments, seg_len, *act_shape)

    obs = obs.reshape(new_obs_shape)
    acts = acts.reshape(new_acts_shape)
    next_obs = next_obs.reshape(new_obs_shape)

    segments = [Segment(*seg_data) for seg_data in zip(obs, acts, next_obs)]

    # Number of segments must be even
    if len(segments) % 2 == 1:
        segments = segments[:-1]

    return segments


def get_eval_trajectories_fn(eval_fn):
    def eval_trajectories_fn(trajectories):
        return np.array([eval_fn(trj) for trj in trajectories])

    return eval_trajectories_fn


def get_obs_acts_next_obs(path):
    if isinstance(path, Segment):
        return zip(path.obs, path.acts, path.next_obs)
    else:
        return zip(path.obs[:-1], path.acts, path.obs[1:])


def get_reward_fn_from_model(rn):
    def get_reward_fn(obs, acts, next_obs, unused_steps):
        sess = tf.get_default_session()
        return sess.run(
            rn.reward_output_train,
            feed_dict={rn.obs_ph: obs, rn.act_ph: acts, rn.next_obs_ph: next_obs,},
        )

    return get_reward_fn


def get_value_fn(model, venv):
    if isinstance(model, ActorCriticRLModel):
        return model.value
    else:
        value_matrix = ti_hard_value_fn(venv)
        return (lambda ob : value_matrix[ob])


def value_diff_eval_path_fn(value_fn):
    def eval_fn(path):
        obs_0 = path.obs[0]
        obs_f = path.next_obs[-1]
        return value_fn([obs_f]) - value_fn([obs_0])
    
    return eval_fn

def one_hot(arr, n):
    return np.eye(n)[arr]

def cloning_eval_path_fn(model):
    if isinstance(model, ActorCriticRLModel):
        def eval_fn(path):
            return model.action_probability(path.obs, actions=path.acts)
    else:
        def eval_fn(path):
            expert_actions = np.array([model.predict(ob)[0] for ob in path.obs])
            return np.sum(expert_actions == path.acts)

    return eval_fn

def soft_expert_cloning_eval_fn(expert, venv):
    if hasattr(expert, 'cross_entropy'):
        def eval_fn(path):
            return np.sum([expert.cross_entropy(ob, act) for ob, act in zip(path.obs, path.acts)])
    else:
        policy, _ = ti_soft_value_fn(venv, beta=10)
        def eval_fn(path):
            action_probs = policy[path.obs, path.acts]
            return np.sum(np.log(action_probs))

    return eval_fn

def reward_eval_path_fn(venv):
    env = get_raw_env(venv)
    if hasattr(env, "eval_trajectory_fn"):
        return env.eval_trajectory_fn
    elif hasattr(env, "reward"):
        if hasattr(env, "state_from_ob"):
            state_fn = env.state_from_ob
        else:
            state_fn = None
        return eval_fn_from_reward(env.reward, state_fn=state_fn)
    else:
        raise Exception("No eval_trajectory_fn in environment")


def eval_fn_from_reward(reward_fn, state_fn=None):
    if state_fn is None:
        state_fn = lambda x : x

    def eval_path_fn(path):
        return sum(
            reward_fn(state_fn(ob), ac, state_fn(next_ob))
            for ob, ac, next_ob in get_obs_acts_next_obs(path)
        )

    return eval_path_fn


def preferences_2(
    venv,
    expert=None,
    evaluate_trajectories_fn=None,
    n_pairs_per_batch=50,
    reward_lr=1e-3,
    policy_lr=1e-3,
    policy_epoch_timesteps=200,
    total_timesteps=10000,
    cloning_bonus=True,
    state_only=False,
    callback=None,
    **kwargs,
):
    if callback is None:
        callback = Callback()


    if evaluate_trajectories_fn is None:
        reward_eval_fn = reward_eval_path_fn(venv)
        evaluate_trajectories_fn = get_eval_trajectories_fn(reward_eval_fn)

        if cloning_bonus:
            cloning_eval_fn = soft_expert_cloning_eval_fn(expert, venv)
            eval_fn = lambda path : reward_eval_fn(path) + cloning_eval_fn(path)
            evaluate_trajectories_fn = get_eval_trajectories_fn(eval_fn)

    # Create reward model
    rn = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        theta_units=[32, 32],
        phi_units=[32, 32],
        scale=True,
        state_only=state_only,
    )

    # Create learner from reward model
    reward_fn = get_reward_fn_from_model(rn)
    venv_train = reward_wrapper.RewardVecEnvWrapper(venv, reward_fn)
    policy = PPO2(MlpPolicy, venv_train, learning_rate=policy_lr)

    # Compute trajectory probabilities
    preferences_ph = tf.placeholder(
        shape=(None, 2), dtype=tf.float32, name="preferences",
    )
    num_segments = 2 * tf.shape(preferences_ph)[0]
    rewards_out = tf.reshape(rn.reward_output_train, [num_segments, -1])
    returns_out = tf.reduce_sum(rewards_out, axis=1)
    returns = tf.reshape(returns_out, shape=[-1, 2])
    log_probs = tf.nn.log_softmax(returns, axis=1)

    # Write loss and optimizer op
    loss = (-1) * tf.reduce_sum(log_probs * preferences_ph)
    optimizer = tf.train.AdamOptimizer(learning_rate=reward_lr)
    reward_train_op = optimizer.minimize(loss)

    # Start training
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    num_epochs = int(np.ceil(total_timesteps / policy_epoch_timesteps))

    callback.start(locals(), globals())
    for epoch in range(num_epochs):
        callback.step(locals(), globals())

        trajectories = sample_trajectories(venv, policy, 2 * n_pairs_per_batch)

        segments = get_segments(trajectories)

        seg_returns = evaluate_trajectories_fn(segments)
        seg_returns = seg_returns.reshape(-1, 2)
        preferences = np.stack(
            [
                seg_returns[:, 0] > seg_returns[:, 1],
                seg_returns[:, 1] > seg_returns[:, 0],
            ],
            axis=1,
        )

        obs = np.concatenate([seg.obs for seg in segments])
        acts = np.concatenate([seg.acts for seg in segments])
        next_obs = np.concatenate([seg.next_obs for seg in segments])

        sess.run(
            reward_train_op,
            feed_dict={
                rn.obs_ph: obs,
                rn.act_ph: acts,
                rn.next_obs_ph: next_obs,
                preferences_ph: preferences,
            },
        )

        # policy.set_env(venv_train)  # Possibly redundant?
        policy.learn(total_timesteps=policy_epoch_timesteps)
    epoch += 1

    callback.step(locals(), globals())

    results = {}
    results["reward_model"] = rn
    results["policy"] = policy

    callback.end(locals(), globals())

    return results


