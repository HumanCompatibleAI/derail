import functools
import os
import sys

import numpy as np

from gym.spaces import Discrete, Box

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from airl.algos.irl_trpo import IRLTRPO
from airl.models.imitation_learning import GAIL, AIRLStateAction

from derail.utils import (
    get_raw_env,
    LightweightRLModel,
    sample_trajectories,
)


def to_rllab_trajectories(trajectories, env):
    def to_one_hot(tensor, n):
        return (np.eye(n)[tensor]).astype(np.float32)

    def to_rllab_traj(traj):
        obs = traj.obs[:-1]
        if isinstance(env.observation_space, Discrete):
            obs = to_one_hot(obs, env.observation_space.n)

        acts = traj.acts
        if isinstance(env.action_space, Discrete):
            acts = to_one_hot(acts, env.action_space.n)

        return {
            "observations": obs,
            "actions": acts,
        }

    return [to_rllab_traj(traj) for traj in trajectories]

def fu_irl(
    venv,
    is_airl,
    expert=None,
    expert_venv=None,
    expert_trajectories=None,
    total_timesteps=10000,
    gen_batch_size=200,
    policy_lr=1e-3,
    callback=None,
    **kwargs,
):
    # Disable algorithm's internal prints
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    raw_env = get_raw_env(venv)
    tf_env = TfEnv(GymEnv(env=raw_env, record_video=False, record_log=False))

    if expert_trajectories is None:
        expert_trajectories = sample_trajectories(
            expert_venv, expert, n_episodes=total_timesteps
        )
    expert_trajectories = to_rllab_trajectories(expert_trajectories, venv)

    if is_airl:
        irl_model = AIRLStateAction(
            env_spec=tf_env.spec, expert_trajs=expert_trajectories
        )
        entropy_weight = 1.0
    else:
        irl_model = GAIL(env_spec=tf_env.spec, expert_trajs=expert_trajectories)
        entropy_weight = 0.0

    if isinstance(venv.action_space, Discrete):
        policy = CategoricalMLPPolicy(
            name="policy", env_spec=tf_env.spec, hidden_sizes=(32, 32)
        )
    else:
        policy = GaussianMLPPolicy(
            name="policy", env_spec=tf_env.spec, hidden_sizes=(32, 32)
        )

    num_epochs = int(total_timesteps // gen_batch_size)

    algo = IRLTRPO(
        env=tf_env,
        policy=policy,
        irl_model=irl_model,
        n_itr=num_epochs,
        batch_size=gen_batch_size,
        max_path_length=100,
        discount=0.99,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=entropy_weight,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=tf_env.spec),
    )
    algo.train()

    sys.stdout = old_stdout

    def predict_fn(ob, state=None, deterministic=False):
        act, _ = algo.policy.get_action(ob)
        return act, state

    results = {}
    results["policy"] = LightweightRLModel(predict_fn=predict_fn, env=venv)

    return results


fu_airl = functools.partial(fu_irl, is_airl=True)
fu_gail = functools.partial(fu_irl, is_airl=False)
