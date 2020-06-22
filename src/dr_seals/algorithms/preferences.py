from collections import Counter, defaultdict, deque, namedtuple
import math
import functools

import numpy as np
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from imitation.rewards.reward_net import BasicShapedRewardNet
from imitation.util import reward_wrapper

from dr_seals.utils import (
    get_random_policy,
    sample_trajectories,
    get_raw_env,
)


def preferences(
    venv,
    evaluate_trajectories_fn=None,
    n_pairs_per_batch=50,
    reward_lr=1e-3,
    policy_lr=1e-3,
    policy_epoch_timesteps=200,
    total_timesteps=10000,
    state_only=False,
    callback=None,
    **kwargs,
):

    if evaluate_trajectories_fn is None:
        evaluate_trajectories_fn = get_eval_trajectories_fn(venv)

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
    venv_train = reward_wrapper.RewardVecEnvWrapper(venv, get_reward_fn_from_model(rn))
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
    for epoch in range(num_epochs):
        if callback is not None:
            callback(locals(), globals())

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

    if callback is not None:
        callback(locals(), globals())

    results = {}
    results["reward_model"] = rn
    results["policy"] = policy

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


def get_eval_trajectories_fn(venv):
    eval_fn = get_eval_trajectory_fn_from_env(venv)

    def eval_trajectories_fn(trajectories):
        returns = np.array([eval_fn(trj) for trj in trajectories])
        return returns

    return eval_trajectories_fn


def get_eval_trajectory_fn_from_env(venv):
    env = get_raw_env(venv)
    if hasattr(env, "eval_trajectory_fn"):
        return env.eval_trajectory_fn
    elif hasattr(env, "reward_fn"):
        return get_eval_path_fn_from_reward(env.reward_fn, env.state_from_ob)
    else:
        raise Error("No eval_trajectory_fn in environment")


def get_eval_path_fn_from_reward(reward_fn, state_fn):
    def get_obs_acts_next_obs(path):
        if isinstance(path, Segment):
            return zip(path.obs, path.acts, path.next_obs)
        else:
            return zip(path.obs[:-1], path.acts, path.obs[1:])

    def eval_path_fn(path):
        return sum(
            reward_fn(state_fn(ob), ac, state_fn(next_ob))
            for ob, ac, next_ob in get_obs_acts_next_obs(path)
        )

    return eval_path_fn


def get_reward_fn_from_model(rn):
    def get_reward_fn(obs, acts, next_obs, unused_steps):
        sess = tf.get_default_session()
        return sess.run(
            rn.reward_output_train,
            feed_dict={rn.obs_ph: obs, rn.act_ph: acts, rn.next_obs_ph: next_obs,},
        )

    return get_reward_fn