import argparse
from concurrent import futures
import functools
import itertools
import re
import os

import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv

from dr_seals.utils import (
    get_expert_algo,
    get_random_policy,
    get_timestamp,
    get_hard_mdp_expert,
    monte_carlo_eval_policy,
    ppo_algo,
    tabular_eval_policy,
    train_rl,
)

from dr_seals.envs import *
from dr_seals.algorithms import *


class EvalCallback:
    def __init__(self, max_n_evals=21, min_step_size=1000):
        self.returns = []
        self.last_timesteps = None

        self.max_n_evals = max_n_evals
        self.min_step_size = min_step_size

    def step(self, lcls, glbs):
        total_timesteps = lcls["total_timesteps"]
        step_size = max(total_timesteps // (self.max_n_evals - 1), self.min_step_size)

        timesteps = lcls["timesteps"]
        if (
            self.last_timesteps is not None
            and timesteps - self.last_timesteps < step_size
        ):
            return

        venv = lcls["venv"]
        policy = lcls["policy"]

        rew = monte_carlo_eval_policy(
            policy, venv, n_eval_episodes=1000, deterministic=False
        )

        self.returns.append((timesteps, rew))

        self.last_timesteps = timesteps

    def get_results(self):
        return self.returns


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
    ):
        if expert_env_name is None:
            expert_env_name = env_name
        if expert_kwargs is None:
            expert_kwargs = {}
        if algo_kwargs is None:
            algo_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}

        self.env_name = env_name
        self.expert_env_name = expert_env_name
        self.expert_kwargs = expert_kwargs
        self.algo_kwargs = algo_kwargs

        self.expert_fn = expert_fn

        self.eval_policy_fn = eval_policy_fn
        self.eval_kwargs = dict(n_eval_episodes=100, deterministic=False)
        self.eval_kwargs.update(eval_kwargs)

        self.callback_cls = EvalCallback
        self.callback_cls = None
        self.callback_kwargs = dict()

    def run(self, algo, **algo_kwargs):
        expert_env = gym.make(f"seals/{self.expert_env_name}-v0")
        expert_env = DummyVecEnv([lambda: expert_env])

        total_timesteps = algo_kwargs.get("total_timesteps", None)

        expert_kwargs = self.expert_kwargs.copy()
        if total_timesteps is not None and "total_timesteps" not in expert_kwargs:
            expert_kwargs["total_timesteps"] = total_timesteps
        expert = self.expert_fn(expert_env, **self.expert_kwargs)

        if self.callback_cls is not None:
            callback = self.callback_cls(**self.callback_kwargs)
            callback_fn = callback.step
        else:
            callback = None
            callback_fn = None

        task_results = {}

        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            env = gym.make(f"seals/{self.env_name}-v0")
            env = DummyVecEnv([lambda: env])

            kwargs = self.algo_kwargs.copy()
            kwargs.update(algo_kwargs)

            algo_results = algo(
                env,
                expert=expert,
                expert_venv=expert_env,
                callback=callback_fn,
                **kwargs,
            )
            learned_policy = algo_results["policy"]

            avg_return = self.eval_policy_fn(learned_policy, env, **self.eval_kwargs)
            task_results["return"] = avg_return

        if self.callback_cls is not None:
            task_results["callback"] = callback.get_results()

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
        env_name="InitStateShiftLearner",
        expert_env_name="InitStateShiftExpert",
        expert_fn=get_hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "early_term_pos": SimpleTask(
        env_name="EarlyTermPos", expert_fn=get_early_term_pos_expert,
    ),
    "early_term_neg": SimpleTask(
        env_name="EarlyTermNeg", expert_fn=get_early_term_neg_expert,
    ),
    "evenodd": SimpleTask(env_name="EvenOdd", expert_fn=get_evenodd_expert,),
    "quadratic": SimpleTask(env_name="Quadratic", expert_fn=get_quadratic_expert,),
    "noisy_obs": SimpleTask(env_name="NoisyObs", expert_fn=get_noisyobs_expert,),
    "risky_path": SimpleTask(
        env_name="RiskyPath",
        expert_fn=get_hard_mdp_expert,
        eval_policy_fn=tabular_eval_policy,
    ),
    "random_goal": SimpleTask(env_name="RandomGoal", expert_fn=get_random_goal_expert,),
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
    "fu_gail": fu_gail,
    "fu_airl": fu_airl,
    "preferences": preferences,
    "airl": imitation_airl,
    "expert": get_expert_algo,
    "random": random_algo,
    "ppo": ppo_algo,
}

def run_experiment(task_name, algo_name, seed, *args, **kwargs):
    print(f"[Running] \t{task_name} {algo_name}")

    task = TASKS[task_name]
    algo = ALGOS[algo_name]
    res = task.run(algo, *args, **kwargs)
    res["task"] = task_name
    res["algo"] = algo_name
    res["seed"] = seed
    return res


def is_compatible(task_name, algo_name):
    continuous_tasks = [
        "sort",
        "noisy_obs",
        "evenodd",
        "quadratic",
        "random_goal",
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

    def get_experiments():
        for seed, task_name, algo_name in itertools.product(range(num_seeds), tasks, algos):
            if is_compatible(task_name, algo_name):
                yield task_name, algo_name, seed

    lines = []

    def log_line(line):
        lines.append(line)
        print(f"[Result] \t{line}")

    def log_result(result):
        task = result["task"]
        algo = result["algo"]
        ret = result["return"]
        seed = result["seed"]
        callback_eval = result.get("callback", None)

        if callback_eval is not None:
            for timesteps, avg_ret in callback_eval:
                log_line(f"{task} {algo} {seed} {timesteps} {avg_ret:.2f}")
        else:
            log_line(f"{task} {algo} {ret:.2f}")

    if parallel:
        with futures.ProcessPoolExecutor(max_workers=None) as executor:
            fts = []
            for spec in get_experiments():
                fts.append(executor.submit(run_experiment, *spec))
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
    parser.add_argument('-l', '--logging', default=True, action='store_true')
    parser.add_argument('-nl', '--no_logging', dest='logging', action='store_false')
    args = parser.parse_args()

    eval_algorithms(
        tasks_regex=args.tasks,
        algos_regex=args.algos,
        parallel=args.parallel,
        timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        logging=args.logging,
    )
