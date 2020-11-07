import argparse
from datetime import datetime
from concurrent import futures
import functools
import itertools
import re
import os
import shutil
import time

import tensorflow as tf

# Remove excessive warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.vec_env import DummyVecEnv

from derail.utils import (
    get_ppo,
    get_random_policy,
    monte_carlo_eval_policy,
    tabular_eval_policy,
    train_rl,
)

import gym
import seals
from derail.experts import *
from derail.algorithms import *


def random_algo(env, *args, **kwargs):
    return {"policy": get_random_policy(env), "reward_model": "random"}

def rl_algo(env, *args, policy_fn=get_ppo, total_timesteps=10000, **kwargs):
    policy = train_rl(env, policy_fn=policy_fn, total_timesteps=total_timesteps)
    return {"policy": policy, "reward_model": "groundtruth"}

def expert_algo(env, expert, *args, **kwargs):
    return {"policy": expert, "reward_model": "groundtruth"}


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

def get_full_env_name(name):
    has_version = "-v" in name
    is_seals = True

    if not has_version:
        name = f'{name}-v0'
    if is_seals:
        name = f'seals/{name}'

    return name

class SimpleTask:
    def __init__(
        self,
        env_name,
        expert_env_name=None,
        expert_kwargs=None,
        expert_fn=train_rl,
        eval_policy_fn=monte_carlo_eval_policy,
        eval_kwargs=None,
    ):
        if expert_env_name is None:
            expert_env_name = env_name
        if expert_kwargs is None:
            expert_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}

        self.env_name = env_name
        self.expert_env_name = expert_env_name
        self.expert_kwargs = expert_kwargs

        self.expert_fn = expert_fn

        self.eval_policy_fn = eval_policy_fn
        self.eval_kwargs = dict(n_eval_episodes=100, deterministic=False)
        self.eval_kwargs.update(eval_kwargs)

    def run(self, algo, seed, **algo_kwargs):
        total_timesteps = algo_kwargs.get("total_timesteps", None)

        get_env = lambda name : DummyVecEnv([lambda: gym.make(get_full_env_name(self.expert_env_name))])

        expert_env = get_env(self.expert_env_name)
        expert_kwargs = self.expert_kwargs.copy()
        if total_timesteps is not None and "total_timesteps" not in expert_kwargs:
            expert_kwargs["total_timesteps"] = total_timesteps
        expert = self.expert_fn(expert_env, **expert_kwargs)

        task_results = {}

        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            env = get_env(self.env_name)

            algo_results = algo(
                env,
                expert=expert,
                expert_venv=expert_env,
                **algo_kwargs,
            )
            learned_policy = algo_results["policy"]

            avg_return = self.eval_policy_fn(learned_policy, env, **self.eval_kwargs)
            task_results["return"] = avg_return

        return task_results



TASKS = {
    "Branching": SimpleTask(
        env_name="Branching",
        expert_fn=hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "InitShift": SimpleTask(
        env_name="InitShiftTest",
        expert_env_name="InitShiftTrain",
        expert_fn=hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "EarlyTermPos": SimpleTask(
        env_name="EarlyTermPos", expert_fn=get_early_term_pos_expert,
    ),
    "EarlyTermNeg": SimpleTask(
        env_name="EarlyTermNeg", expert_fn=get_early_term_neg_expert,
    ),
    "LargestSum": SimpleTask(env_name="LargestSum", expert_fn=get_largest_sum_expert,),
    "Parabola": SimpleTask(env_name="Parabola", expert_fn=get_parabola_expert,),
    "NoisyObs": SimpleTask(env_name="NoisyObs", expert_fn=get_noisyobs_expert,),
    "RiskyPath": SimpleTask(
        env_name="RiskyPath",
        expert_fn=hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "ProcGoal": SimpleTask(env_name="ProcGoal", expert_fn=get_proc_goal_expert,),
    "Sort": SimpleTask(env_name="Sort", expert_fn=get_selectionsort_expert,),
}


ALGOS = {
    "expert": expert_algo,
    "random": random_algo,
    "ppo": rl_algo,
    "bc": behavioral_cloning,
    "gail_im": imitation_gail,
    "gail_sb": stable_gail,
    "gail_fu": fu_gail,
    "airl_fu": fu_airl,
    "airl_im_sa": imitation_airl,
    "airl_im_so": functools.partial(imitation_airl, state_only=True),
    "drlhp_sa": preferences,
    "drlhp_so": functools.partial(preferences, state_only=True),
    "maxent_irl": max_ent_irl,
    "mce_irl": mce_irl,
    "drlhp_slow": functools.partial(preferences, policy_lr=1e-4),
    "drlhp_eps": functools.partial(preferences, egreedy_sampling=True),
    "drlhp_rnd": functools.partial(preferences, use_rnd_bonus=True),
}


def run_experiment(task_name, algo_name, seed, *args, **kwargs):
    print(f"[Running] \t{task_name} {algo_name}")

    task = TASKS[task_name]
    algo = ALGOS[algo_name]
    res = task.run(algo, *args, seed=seed, algo_name=algo_name, **kwargs)
    res["task"] = task_name
    res["algo"] = algo_name
    res["seed"] = seed
    return res


def is_compatible(task_name, algo_name):
    continuous_tasks = [
        "Sort",
        "NoisyObs",
        "LargestSum",
        "Parabola",
        "ProcGoal",
    ]
    needs_discrete_algos = ["maxent_irl", "mce_irl"]
    if algo_name in needs_discrete_algos and any(
        pattern in task_name for pattern in continuous_tasks
    ):
        return False

    variable_horizon_tasks = ["EarlyTerm"]
    fixed_horizon_algos = ["max_ent_irl", "mce_irl"]
    if algo_name in fixed_horizon_algos and any(
        pattern in task_name for pattern in variable_horizon_tasks
    ):
        return False

    has_fu_conflict = [
        "Sort", # uses MultiDiscrete, not supported
    ]
    fu_algos = [
        "airl_fu",
        "gail_fu",
    ]
    if algo_name in fu_algos and any(
        pattern in task_name for pattern in has_fu_conflict
    ):
        return False

    return True


def eval_algorithms(
    tasks_regex,
    algos_regex,
    parallel=False,
    timesteps=10000,
    num_seeds=1,
    logging=True,
    **kwargs,
):
    timestamp = get_timestamp()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tasks = [name for name in TASKS if re.search(tasks_regex, name) is not None]
    algos = [name for name in ALGOS if re.search(algos_regex, name) is not None]

    print("*** ALGORITHMS ***", *algos, sep="\n", end="\n\n")
    print("*** TASKS ***", *tasks, sep="\n", end="\n\n")

    def get_experiments():
        for seed, task_name, algo_name in itertools.product(
            range(num_seeds), tasks, algos,
        ):
            if is_compatible(task_name, algo_name):
                yield task_name, algo_name, seed

    lines = []

    os.makedirs('results/partial/', exist_ok=True)
    def log_line(line):
        lines.append(line)
        print(f"[Result] \t{line}")
        with open(f"results/partial/{timestamp}.csv", "a") as f:
            f.write(f"{line}\n")

    def log_result(result):
        if not logging:
            return 

        task = result["task"]
        algo = result["algo"]
        ret = result["return"]
        seed = result["seed"]

        include_seed = False
        if include_seed:
            maybe_seed = f" {seed}"
        else:
            maybe_seed = ""

        log_line(f"{task} {algo}{maybe_seed} {ret:.2f}")

    if parallel:
        with futures.ProcessPoolExecutor(max_workers=None) as executor:
            fts = []
            for spec in get_experiments():
                fts.append(
                    executor.submit(run_experiment, *spec, total_timesteps=timesteps)
                )
                time.sleep(0.01)
            for f in futures.as_completed(fts):
                log_result(f.result())
    else:
        for spec in get_experiments():
            res = run_experiment(*spec, total_timesteps=timesteps)
            log_result(res)

    if logging:
        lines.sort()

        os.makedirs('results', exist_ok=True)
        results_file = f"results/{timestamp}.csv"
        with open(results_file, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")
        shutil.copy(results_file, 'results/last.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diagnostic task suite")
    parser.add_argument(
        "--tasks", "-e", type=str, default=r".*", help="regex of tasks to run"
    )
    parser.add_argument(
        "--algos", "-a", type=str, default=r".*", help="regex of algos to run"
    )
    parser.add_argument(
        "--num_seeds",
        "-n",
        type=int,
        default=1,
        help="Number of seeds for each experiment",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=10000,
        help="Number of timesteps for algorithms",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        default=False,
        action="store_true",
        help="Whether to run experiments in parallel",
    )
    parser.add_argument("-l", "--logging", default=True, action="store_true")
    parser.add_argument("-nl", "--no_logging", dest="logging", action="store_false")
    args = parser.parse_args()

    eval_algorithms(
        tasks_regex=args.tasks,
        algos_regex=args.algos,
        parallel=args.parallel,
        timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        logging=args.logging,
    )
