import json

import itertools
from collections import Counter, defaultdict
import os
import pickle

import numpy as np

from derail.utils import get_horizon, get_raw_policy, get_raw_env, sample_trajectories, RunningMeanVar

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
        num_calls = max(num_calls, 1)

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




class CorridorDrlhpCallback:
    def __init__(self, savepath, max_num_calls=100, min_timesteps=1000):
        self.savepath = savepath
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

        self._queried = Counter()


    def step(self, lcls, gbls, idx=0):
        epoch = lcls['epoch']
        num_epochs = lcls['num_epochs']

        if epoch % self._epoch_multiple == 0 or epoch == num_epochs:
            self.data.append(self.get_snapshot(lcls, gbls, idx))

        self._call_idx += 1

    def get_snapshot(self, lcls, gbls, idx=0):
        venv = lcls['venv']
        policy = lcls['policy']
        timesteps = lcls['policy_epoch_timesteps'] * lcls['epoch']
        use_rnd = lcls['use_rnd']
        rew_fn = lcls['reward_fn']
        rnd_reward_fn = lcls.get('rnd_reward_fn', lambda _ : 0)
        obs = lcls.get('obs', [])
        runn_rnd_rews = lcls.get('runn_rnd_rews', RunningMeanVar())
        runn_ext_rews = lcls.get('runn_ext_rews', RunningMeanVar())

        self._queried.update(obs)

        P = get_raw_policy(policy)[0]

        nS = venv.observation_space.n
        nA = venv.action_space.n
        S = range(nS)
        A = range(nA)

        env = get_raw_env(venv)

        horizon = get_horizon(venv)

        def get_occupancy(pol):
            occupancy = np.zeros((horizon+1, nS))
            occupancy[0, 0] = 1.0

            for t in range(horizon):
                for s, a in itertools.product(S, A):
                    ns = env.transition(s, a)
                    occupancy[t + 1, ns] += occupancy[t, s] * P[s, a]

            return occupancy.sum(axis=0) / occupancy.sum()


        # Turn off reward running means (side effects)
        runn_rnd_rews.update_on = False
        runn_ext_rews.update_on = False

        R = np.zeros((nS, nA))
        for s, a in itertools.product(S, A):
            ns = env.transition(s, a)
            R[s, a] = rew_fn([s], [a], [ns], None)

        Rint = np.array([rnd_reward_fn([s]) for s in S])

        runn_rnd_rews.update_on = True
        runn_ext_rews.update_on = True

        O = get_occupancy(P)

        fmt_float = lambda x : round(float(x), 2)



        def pos_info(s):
            return {
                'rew_l' : fmt_float(R[s, 0] - Rint[s]),
                'rew_r' : fmt_float(R[s, 1] - Rint[s]),
                'rew_rnd' : fmt_float(Rint[s]),
                'occ' : fmt_float(O[s]),
                'queried' : fmt_float(self._queried.get(s, 0)),
            }

        return {
            "timesteps" : int(timesteps),
            "states" : [pos_info(s) for s in S],
        }


    def end(self, lcls, gbls):
        dirpath = os.path.dirname(self.savepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(f'{self.savepath}.pkl', 'wb') as f:
            pickle.dump(self.data, f)
        with open(f'{self.savepath}.json', 'w') as f:
            json.dump(self.data, f, indent=2)



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


