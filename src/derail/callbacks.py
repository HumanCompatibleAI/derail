import json

import itertools
from collections import Counter, defaultdict
import os
import pickle

import numpy as np

from derail.utils import get_raw_env, sample_trajectories

class Callback:
    def start(self, lcls, gbls):
        pass

    def step(self, lcls, gbls, idx=0):
        pass

    def end(self, lcls, gbls):
        pass


class TfCollectorCallback:
    def __init__(self, savepath):
        self.savepath = savepath
        os.makedirs(self.savepath, exist_ok=True)

        self._call_idx = 0

    def start(self, lcls, gbls):
        import tensorflow as tf
        self._saver = tf.train.Saver(max_to_keep=None)

        policy = lcls['policy']
        p_sess = policy.sess

    def step(self, lcls, gbls, idx=0):
        policy = lcls['policy']
        p_sess = policy.sess
        sess = lcls['sess']

        self._saver.save(sess, f'{self.savepath}-{str(self._call_idx)}')
        self._saver.save(sess, f'{self.savepath}-{str(self._call_idx)}')
        self._call_idx += 1

    def end(self, lcls, gbls):
        pass



def noisy_obs_extractor(venv, policy, reward, info):
    n_samples = 30
    n_episodes = 100

    env = get_raw_env(venv)

    def precompute_occupancies():
        trajectories = sample_trajectories(venv, policy, n_episodes=n_episodes)

        g_pos = lambda ob : tuple(ob[:2])

        obs = np.concatenate([trj.obs for trj in trajectories])

        c = Counter(g_pos(ob) for ob in obs)
        m = n_episodes * len(trajectories[0].obs)

        occ = defaultdict(int)
        occ.update({k : v / m for k, v in c.items()})
        return occ

    occ = precompute_occupancies()

    def estimate_occupancy(pos):
        return occ[pos]


    def estimate_reward(pos):
        nl = 20

        def sample_rew(pos):
            pos = np.array(pos)
            ob = env.obs_from_state(pos)
            act = policy.predict(ob)[0]
            next_pos = env.transition(pos, act)
            next_ob = env.obs_from_state(next_pos)

            return reward([ob], [act], [next_ob], None)

        return np.mean([sample_rew(pos) for _ in range(n_samples)])


    def pos_info(pos):
        return {
            'rew' : round(float(estimate_reward(pos)), 2),
            'occ' : round(float(estimate_occupancy(pos)), 2),
        }

    L = range(5)
    states_data = [pos_info(pos) for pos in itertools.product(L, L)]


    return {
        "timesteps" : int(info['timesteps']),
        "states" : states_data,
    }


def drlhp_extractor(lcls, gbls):
    venv = lcls['venv']
    policy = lcls['policy']
    reward_fn = lcls['reward_fn']

    info = {}
    info['timesteps'] = lcls['policy_epoch_timesteps'] * lcls['epoch']

    return venv, policy, reward_fn, info


class CollectorCallback:
    def __init__(self, savepath, algo_xfn, env_xfn, max_num_calls=100, min_timesteps=1000):
        self.savepath = savepath
        self.algo_xfn = algo_xfn
        self.env_xfn = env_xfn
        self.max_num_calls = max_num_calls
        self.min_timesteps = min_timesteps

    def start(self, lcls, gbls):
        self._call_idx = 0
        self.data = []

        total_timesteps = lcls['total_timesteps'] 
        policy_epoch_timesteps = lcls['policy_epoch_timesteps']
        num_epochs = lcls['num_epochs']

        num_calls = min(self.max_num_calls, total_timesteps // self.min_timesteps)

        self._epoch_multiple = num_epochs // num_calls


    def step(self, lcls, gbls, idx=0):
        epoch = lcls['epoch']
        num_epochs = lcls['num_epochs']

        if epoch % self._epoch_multiple == 0 or epoch == num_epochs:
            env, policy, reward, info = self.algo_xfn(lcls, gbls)
            step_data = self.env_xfn(env, policy, reward, info)
            self.data.append(step_data)

        self._call_idx += 1

    def end(self, lcls, gbls):
        dirpath = os.path.dirname(self.savepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(f'{self.savepath}.pkl', 'wb') as f:
            pickle.dump(self.data, f)
        with open(f'{self.savepath}.json', 'w') as f:
            json.dump(self.data, f)




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


