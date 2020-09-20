import uuid

# from stable_baselines import GAIL2
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from derail.utils import (
    get_horizon,
    get_ppo,
    get_raw_env,
    LightweightRLModel,
    sample_trajectories,
)


def get_expert_dataset(
    expert, venv, total_timesteps,
):

    filename = f"/tmp/{uuid.uuid4()}"
    n_episodes = total_timesteps // get_horizon(venv)

    generate_expert_traj(expert, save_path=filename, env=venv, n_episodes=n_episodes)
    dataset = ExpertDataset(expert_path=f"{filename}.npz", verbose=0)

    return dataset


def behavioral_cloning(
    venv,
    expert=None,
    expert_venv=None,
    expert_trajectories=None,
    state_only=False,
    policy_fn=get_ppo,
    total_timesteps=10000,
    policy_lr=1e-3,
    callback=None,
    **kwargs,
):
    dataset = get_expert_dataset(expert, expert_venv, total_timesteps)

    policy = get_ppo(venv, learning_rate=policy_lr)
    policy.pretrain(dataset)

    results = {}
    results["policy"] = policy

    return results


def stable_gail(
    venv,
    expert=None,
    expert_venv=None,
    state_only=False,
    total_timesteps=10000,
    gen_batch_size=200,
    disc_batch_size=100,
    policy_lr=1e-3,
    callback=None,
    **kwargs,
):
    dataset = get_expert_dataset(expert, expert_venv, total_timesteps)

    policy = GAIL("MlpPolicy", venv, dataset)
    policy.learn(total_timesteps=total_timesteps)

    results = {}
    results["policy"] = policy

    return results

def stable_gail_2(
    venv,
    expert=None,
    expert_venv=None,
    state_only=False,
    total_timesteps=10000,
    gen_batch_size=200,
    disc_batch_size=100,
    policy_lr=1e-3,
    callback=None,
    **kwargs,
):
    dataset = get_expert_dataset(expert, expert_venv, total_timesteps)

    policy = GAIL2("MlpPolicy", venv, dataset)
    policy.learn(total_timesteps=total_timesteps)

    results = {}
    results["policy"] = policy

    return results
