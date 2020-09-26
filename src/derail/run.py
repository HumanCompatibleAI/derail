import argparse
from concurrent import futures
import functools
import itertools
import re
import os

import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv

from derail.utils import (
    get_expert_algo,
    get_hard_mdp_expert,
    get_ppo,
    get_random_policy,
    get_last_timestamp,
    get_timestamp,
    monte_carlo_eval_policy,
    ppo_algo,
    tabular_eval_policy,
    train_rl,
)

from derail.envs import *
from derail.envs.experts import *
from derail.algorithms import *
from derail.callbacks import *


def name_with_version(name):
    if "-v" in name:
        return name
    else:
        return f"{name}-v0"


class SimpleTask:
    def __init__(
        self,
        env_name,
        expert_env_name=None,
        expert_kwargs=None,
        algo_kwargs=None,
        expert_fn=train_rl,
        eval_policy_fn=monte_carlo_eval_policy,
        eval_kwargs=None,
        callback_cls=Callback,
        callback_kwargs=None,
    ):
        if expert_env_name is None:
            expert_env_name = env_name
        if expert_kwargs is None:
            expert_kwargs = {}
        if algo_kwargs is None:
            algo_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}
        if callback_kwargs is None:
            callback_kwargs = {}

        self.env_name = env_name
        self.expert_env_name = expert_env_name
        self.expert_kwargs = expert_kwargs
        self.algo_kwargs = algo_kwargs

        self.expert_fn = expert_fn

        self.eval_policy_fn = eval_policy_fn
        self.eval_kwargs = dict(n_eval_episodes=100, deterministic=False)
        self.eval_kwargs.update(eval_kwargs)

        self.callback_cls = callback_cls
        self.callback_kwargs = callback_kwargs

    def run(self, algo, seed, **algo_kwargs):
        # XXX Hack
        self.callback_cls = CollectorCallback

        algo_name = algo_kwargs['algo_name']

        savepath = os.path.join(
            './data',
            get_last_timestamp(),
            f'{self.env_name}-{algo_name}-{seed}',
        )

        self.callback_kwargs = dict(savepath=savepath)


        expert_env = gym.make(f"seals/{name_with_version(self.expert_env_name)}")
        expert_env = DummyVecEnv([lambda: expert_env])

        total_timesteps = algo_kwargs.get("total_timesteps", None)

        expert_kwargs = self.expert_kwargs.copy()
        if total_timesteps is not None and "total_timesteps" not in expert_kwargs:
            expert_kwargs["total_timesteps"] = total_timesteps
        expert = self.expert_fn(expert_env, **expert_kwargs)

        # callback = self.callback_cls(**self.callback_kwargs)

        # XXX Hack
        # XXX Hack
        if 'Noisy' in self.env_name and 'pref' in algo_name:
            callback = CollectorCallback(savepath, algo_xfn=drlhp_extractor, env_xfn=noisy_obs_extractor)
        elif 'orridor' in self.env_name and 'pref' in algo_name:
            callback = CorridorDrlhpCallback(savepath)
        else:
            callback = Callback()

        task_results = {}

        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            env = gym.make(f"seals/{name_with_version(self.env_name)}")
            env = DummyVecEnv([lambda: env])

            kwargs = self.algo_kwargs.copy()
            kwargs.update(algo_kwargs)

            algo_results = algo(
                env,
                expert=expert,
                expert_venv=expert_env,
                callback=callback,
                **kwargs,
            )
            learned_policy = algo_results["policy"]

            avg_return = self.eval_policy_fn(learned_policy, env, **self.eval_kwargs)
            task_results["return"] = avg_return

        return task_results


def random_algo(env, *args, **kwargs):
    return {"policy": get_random_policy(env), "reward_model": "random"}


TASKS = {
    "branching": SimpleTask(
        env_name="Branching",
        expert_fn=get_hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "init_state_shift": SimpleTask(
        env_name="InitShiftTest",
        expert_env_name="InitShiftTrain",
        expert_fn=get_hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "early_term_pos": SimpleTask(
        env_name="EarlyTermPos", expert_fn=get_early_term_pos_expert,
    ),
    "early_term_neg": SimpleTask(
        env_name="EarlyTermNeg", expert_fn=get_early_term_neg_expert,
    ),
    "largest_sum": SimpleTask(env_name="LargestSum", expert_fn=get_largest_sum_expert,),
    "parabola": SimpleTask(env_name="Parabola", expert_fn=get_parabola_expert,),
    "noisy_obs": SimpleTask(env_name="NoisyObs", expert_fn=get_noisyobs_expert,),
    "noisy_obs": SimpleTask(env_name="NoisyObs", expert_fn=get_noisyobs_expert,),
    "corridor_v0": SimpleTask(env_name="Corridor-v0", expert_fn=get_corridor_expert,),
    "corridor_v1": SimpleTask(env_name="Corridor-v1", expert_fn=get_corridor_expert,),
    "corridor_v2": SimpleTask(env_name="Corridor-v2", expert_fn=get_corridor_expert,),
    "corridor_v3": SimpleTask(env_name="Corridor-v3", expert_fn=get_corridor_expert,),
    "corridor_v4": SimpleTask(env_name="Corridor-v4", expert_fn=get_corridor_expert,),
    "corridor_v5": SimpleTask(env_name="Corridor-v5", expert_fn=get_corridor_expert,),
    "corridor_v6": SimpleTask(env_name="Corridor-v6", expert_fn=get_corridor_expert,),
    "corridor_v7": SimpleTask(env_name="Corridor-v7", expert_fn=get_corridor_expert,),
    "corridor_v8": SimpleTask(env_name="Corridor-v8", expert_fn=get_corridor_expert,),
    # "noisy_obs_v1": SimpleTask(env_name="NoisyObs-v1", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v1": SimpleTask(env_name="NoisyObs-v1", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v1": SimpleTask(env_name="NoisyObs-v1", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v2": SimpleTask(env_name="NoisyObs-v2", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v3": SimpleTask(env_name="NoisyObs-v3", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v4": SimpleTask(env_name="NoisyObs-v4", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v5": SimpleTask(env_name="NoisyObs-v5", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v6": SimpleTask(env_name="NoisyObs-v6", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v7": SimpleTask(env_name="NoisyObs-v7", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v8": SimpleTask(env_name="NoisyObs-v8", expert_fn=get_noisyobs_expert,),
    # "noisy_obs_v9": SimpleTask(env_name="NoisyObs-v9", expert_fn=get_noisyobs_expert,),
    "risky_path": SimpleTask(
        env_name="RiskyPath",
        expert_fn=get_hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "proc_goal": SimpleTask(env_name="ProcGoal", expert_fn=get_proc_goal_expert,),
    "sort": SimpleTask(env_name="Sort", expert_fn=get_selectionsort_expert,),
}


ALGOS = {
    "mce_irl": mce_irl,
    "max_ent_irl": max_ent_irl,
    "airl_state_only": functools.partial(imitation_airl, state_only=True),
    "preferences_state_only": functools.partial(preferences, state_only=True),
    "imitation_gail": imitation_gail,
    "behavioral_cloning": behavioral_cloning,
    "stable_gail": stable_gail,
    # "stable_gail_2": stable_gail_2,
    "fu_gail": fu_gail,
    "fu_airl": fu_airl,
    "preferences": preferences,
    # "preferences_rnd": functools.partial(preferences, use_rnd=True),
    "airl": imitation_airl,

    "expert": get_expert_algo,
    "random": random_algo,
    "ppo": ppo_algo,
}

for i, args in enumerate(itertools.product(
    (False, True), # cloning_bonus
    (False, True), # use_rnd
    (False, True), # egreedy_sampling
    (1e-3, 1e-4), # policy_lr
)):
    cloning_bonus, use_rnd, egreedy_sampling, policy_lr = args

    ALGOS[f'preferences_{i:04b}'] = functools.partial(
        preferences,
        cloning_bonus=cloning_bonus,
        use_rnd=use_rnd,
        egreedy_sampling=egreedy_sampling,
        policy_lr=policy_lr,
    )


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
        "sort",
        "noisy_obs",
        "largest_sum",
        "parabola",
        "proc_goal",
    ]
    needs_discrete_algos = ["max_ent_irl", "mce_irl"]
    if algo_name in needs_discrete_algos and any(
        pattern in task_name for pattern in continuous_tasks
    ):
        return False

    variable_horizon_tasks = ["early_term"]
    fixed_horizon_algos = ["max_ent_irl", "mce_irl"]
    if algo_name in fixed_horizon_algos and any(
        pattern in task_name for pattern in variable_horizon_tasks
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
            range(num_seeds), tasks, algos
        ):
            if is_compatible(task_name, algo_name):
                yield task_name, algo_name, seed

    lines = []

    def log_line(line):
        lines.append(line)
        print(f"[Result] \t{line}")
        with open(f"partial-results-{timestamp}.csv", "a") as f:
            f.write(f"{line}\n")

    def log_result(result):
        task = result["task"]
        algo = result["algo"]
        ret = result["return"]
        seed = result["seed"]
        callback_eval = result.get("callback", None)

        include_seed = True
        if include_seed:
            maybe_seed = f" {seed}"
        else:
            maybe_seed = ""

        if callback_eval is not None:
            for timesteps, avg_ret in callback_eval:
                log_line(f"{task} {algo} {seed} {timesteps} {avg_ret:.2f}")
        else:
            log_line(f"{task} {algo}{maybe_seed} {ret:.2f}")

    if parallel:
        with futures.ProcessPoolExecutor(max_workers=None) as executor:
            fts = []
            for spec in get_experiments():
                fts.append(
                    executor.submit(run_experiment, *spec, total_timesteps=timesteps)
                )
            for f in futures.as_completed(fts):
                log_result(f.result())
    else:
        for spec in get_experiments():
            res = run_experiment(*spec, total_timesteps=timesteps)
            log_result(res)

    if logging:
        lines.sort()
        with open(f"results-{timestamp}.csv", "w") as f:
            f.write("\n".join(lines))
            f.write("\n")


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
