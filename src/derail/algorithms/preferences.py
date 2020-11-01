from collections import Counter, defaultdict, deque, namedtuple
import collections

import math
import functools

import numpy as np
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines import PPO2
from stable_baselines.common.base_class import ActorCriticRLModel
from stable_baselines.common.policies import MlpPolicy

from imitation.rewards.reward_net import BasicShapedRewardNet
from imitation.util import reward_wrapper

from derail.utils import (
    get_random_policy,
    sample_trajectories,
    get_raw_env,
    get_horizon,
    make_egreedy,
    ti_hard_value_fn,
    ti_soft_value_fn,
    RunningMeanVar,
)

def preferences(
    venv,
    expert=None,
    evaluate_trajectories_fn=None,
    n_pairs_per_batch=50,
    n_timesteps_per_query=None,
    reward_lr=1e-3,
    policy_lr=1e-3,
    policy_epoch_timesteps=200,
    total_timesteps=10000,
    state_only=False,
    use_rnd=False,
    rnd_lr=1e-3,
    rnd_coeff=0.5,
    normalize_extrinsic=False,
    egreedy_sampling=False,
    **kwargs,
):
    if n_pairs_per_batch is None:
        horizon = get_horizon(venv)
        n_pairs_per_batch = (n_timesteps_per_query / (2 * horizon))


    if evaluate_trajectories_fn is None:
        reward_eval_fn = reward_eval_path_fn(venv)
        evaluate_trajectories_fn = get_eval_trajectories_fn(reward_eval_fn)

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
    if use_rnd:
        rnd_size = 50

        inputs = [rn.obs_inp, rn.act_inp]
        inputs = [tf.layers.flatten(x) for x in inputs]
        inputs = tf.concat(inputs, axis=1)

        rnd_target_net = build_mlp([32, 32, 32], output_size=rnd_size)
        rnd_target = sequential(inputs, rnd_target_net)

        rnd_pred_net = build_mlp([32, 32, 32], output_size=rnd_size)
        rnd_pred = sequential(inputs, rnd_pred_net)

        rnd_loss = tf.reduce_mean((tf.stop_gradient(rnd_target) - rnd_pred)**2)
        rnd_optimizer = tf.train.AdamOptimizer(learning_rate=rnd_lr)
        rnd_train_op = rnd_optimizer.minimize(rnd_loss)

        runn_rnd_rews = RunningMeanVar(alpha=0.01)

        def rnd_reward_fn(obs, acts=None, *args, **kwargs):
            if acts is None:
                acts = [venv.action_space.sample()]
            int_rew = sess.run(rnd_loss, feed_dict={rn.obs_ph : obs, rn.act_ph: acts})
            int_rew_old = int_rew
            int_rew = runn_rnd_rews.exp_update(int_rew)

            return int_rew

        base_extrinsic_reward_fn = reward_fn

        if normalize_extrinsic:
            runn_ext_rews = RunningMeanVar(alpha=0.01)

        def extrinsic_reward_fn(*args, **kwargs):
            ext_rew = base_extrinsic_reward_fn(*args, **kwargs)
            if normalize_extrinsic:
                ext_rew = runn_ext_rews.exp_update(ext_rew)
            return ext_rew

        def reward_fn(*args, **kwargs):
            return extrinsic_reward_fn(*args, **kwargs) + rnd_coeff * rnd_reward_fn(*args, **kwargs)

    # Create learner from reward model
    venv_train = reward_wrapper.RewardVecEnvWrapper(venv, reward_fn)
    policy = PPO2(MlpPolicy, venv_train, learning_rate=policy_lr)


    # Start training
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    sampling_policy = make_egreedy(policy, venv) if egreedy_sampling else policy

    num_epochs = int(np.ceil(total_timesteps / policy_epoch_timesteps))

    for epoch in range(num_epochs):
        trajectories = sample_trajectories(venv, sampling_policy, 2 * n_pairs_per_batch)

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

        ops = [reward_train_op]
        if use_rnd:
            ops.append(rnd_train_op)

        sess.run(
            ops,
            feed_dict={
                rn.obs_ph: obs,
                rn.act_ph: acts,
                rn.next_obs_ph: next_obs,
                preferences_ph: preferences,
            },
        )

        policy.learn(total_timesteps=policy_epoch_timesteps)

    results = {}
    results["reward_model"] = rn
    results["policy"] = policy

    return results


Segment = namedtuple("Segment", ["obs", "acts", "next_obs"])

def build_mlp(hid_sizes,
              output_size=1,
              name=None,
              activation=tf.nn.relu,
              initializer=None,
              ):
  """Constructs an MLP, returning an ordered dict of layers."""
  layers = collections.OrderedDict()

  # Hidden layers
  for i, size in enumerate(hid_sizes):
    key = f"{name}_dense{i}"
    layer = tf.layers.Dense(size, activation=activation,
                            kernel_initializer=initializer,
                            name=key)  # type: tf.layers.Layer
    layers[key] = layer

  # Final layer
  layer = tf.layers.Dense(output_size, kernel_initializer=initializer,
                          name=f"{name}_dense_final")  # type: tf.layers.Layer
  layers[f"{name}_dense_final"] = layer

  return layers

def sequential(inputs,
               layers,
               ):
  """Applies a sequence of layers to an input."""
  output = inputs
  for layer in layers.values():
    output = layer(output)
  return output


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
