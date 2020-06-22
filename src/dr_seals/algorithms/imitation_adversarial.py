import functools

import numpy as np
import tensorflow as tf

from imitation.rewards.discrim_net import DiscrimNetAIRL, DiscrimNetGAIL
from imitation.rewards.reward_net import BasicShapedRewardNet

from imitation.util.buffering_wrapper import BufferingWrapper
from imitation.util import buffer, reward_wrapper
from imitation.util.rollout import flatten_trajectories

from dr_seals.utils import (
    get_ppo,
    sample_trajectories,
)

def adversarial_learning(
    venv,
    expert=None,
    expert_venv=None,
    expert_trajectories=None,
    state_only=False,
    policy_fn=get_ppo,
    total_timesteps=20000,
    gen_batch_size=200,
    disc_batch_size=100,
    updates_per_batch=2,
    policy_lr=1e-3,
    reward_lr=1e-3,
    is_airl=True,
    callback=None,
    **kwargs,
):
    # Set up generator
    gen_policy = policy_fn(venv, learning_rate=policy_lr)
    policy = gen_policy

    # Set up discriminator
    if is_airl:
        rn = BasicShapedRewardNet(
            venv.observation_space,
            venv.action_space,
            theta_units=[32, 32],
            phi_units=[32, 32],
            scale=True,
            state_only=state_only,
        )
        discrim = DiscrimNetAIRL(rn, entropy_weight=1.0)
    else:
        rn = None
        discrim = DiscrimNetGAIL(venv.observation_space, venv.action_space)

    # Set up optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=reward_lr).minimize(
        tf.reduce_mean(discrim.disc_loss)
    )

    # Set up environment reward
    reward_train = functools.partial(
        discrim.reward_train, gen_log_prob_fn=gen_policy.action_probability
    )
    venv_train = reward_wrapper.RewardVecEnvWrapper(venv, reward_train)
    venv_train_buffering = BufferingWrapper(venv_train)
    gen_policy.set_env(venv_train_buffering)  # possibly redundant

    # Set up replay buffers
    gen_replay_buffer_capacity = 20 * gen_batch_size
    gen_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity, venv)

    if expert_trajectories is not None:
        expert_transitions = flatten_trajectories(expert_trajectories)
        exp_replay_buffer = buffer.ReplayBuffer.from_data(expert_transitions)
    else:
        exp_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity, venv)

    # Start training
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    num_epochs = int(np.ceil(total_timesteps / gen_batch_size))
    for epoch in range(num_epochs):
        timesteps = gen_batch_size * epoch
        if callback is not None:
            callback(locals(), globals())

        # Train gen
        # gen_policy.set_env(venv_train_buffering)  # possibly redundant
        gen_policy.learn(total_timesteps=gen_batch_size, reset_num_timesteps=True)
        gen_replay_buffer.store(venv_train_buffering.pop_transitions())

        if expert_trajectories is None:
            exp_replay_buffer.store(
                flatten_trajectories(
                    sample_trajectories(expert_venv, expert, n_timesteps=gen_batch_size)
                )
            )

        # Train disc
        for _ in range(updates_per_batch):
            disc_minibatch_size = disc_batch_size // updates_per_batch
            half_minibatch = disc_minibatch_size // 2

            gen_samples = gen_replay_buffer.sample(half_minibatch)
            expert_samples = exp_replay_buffer.sample(half_minibatch)

            obs = np.concatenate([gen_samples.obs, expert_samples.obs])
            acts = np.concatenate([gen_samples.acts, expert_samples.acts])
            next_obs = np.concatenate([gen_samples.next_obs, expert_samples.next_obs])
            labels = np.concatenate(
                [np.ones(half_minibatch), np.zeros(half_minibatch)]
            )

            log_act_prob = gen_policy.action_probability(obs, actions=acts, logp=True)
            log_act_prob = log_act_prob.reshape((disc_minibatch_size,))

            _, logits_v, loss_v = sess.run(
                [train_op, discrim._disc_logits_gen_is_high, discrim._disc_loss,],
                feed_dict={
                    discrim.obs_ph: obs,
                    discrim.act_ph: acts,
                    discrim.next_obs_ph: next_obs,
                    discrim.labels_gen_is_one_ph: labels,
                    discrim.log_policy_act_prob_ph: log_act_prob,
                },
            )

    timesteps = total_timesteps
    if callback is not None:
        callback(locals(), globals())

    results = {}
    results["reward_model"] = rn
    results["discrim"] = discrim
    results["policy"] = gen_policy

    return results


imitation_airl = functools.partial(adversarial_learning, is_airl=True)
imitation_gail = functools.partial(adversarial_learning, is_airl=False)
