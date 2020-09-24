"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/Branching-v0",
    entry_point="derail.envs.branching:BranchingEnv",
    max_episode_steps=11,
)

gym.register(
    id="seals/EarlyTermNeg-v0",
    entry_point="derail.envs.early_term:EarlyTermNegEnv",
    max_episode_steps=10,
)

gym.register(
    id="seals/EarlyTermPos-v0",
    entry_point="derail.envs.early_term:EarlyTermPosEnv",
    max_episode_steps=10,
)

gym.register(
    id="seals/InitShiftTrain-v0",
    entry_point="derail.envs.init_shift:InitShiftTrainEnv",
    max_episode_steps=3,
)

gym.register(
    id="seals/InitShiftTest-v0",
    entry_point="derail.envs.init_shift:InitShiftTestEnv",
    max_episode_steps=3,
)

gym.register(
    id="seals/LargestSum-v0",
    entry_point="derail.envs.largest_sum:LargestSumEnv",
    max_episode_steps=1,
)

gym.register(
    id="seals/NoisyObs-v0",
    entry_point="derail.envs.noisy_obs:NoisyObsEnv",
    max_episode_steps=15,
)

gym.register(
    id="seals/Parabola-v0",
    entry_point="derail.envs.parabola:ParabolaEnv",
    max_episode_steps=20,
)

gym.register(
    id="seals/ProcGoal-v0",
    entry_point="derail.envs.proc_goal:ProcGoalEnv",
    max_episode_steps=20,
)

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="derail.envs.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/Sort-v0",
    entry_point="derail.envs.sort:SortEnv",
    max_episode_steps=6,
)



gym.register(
    id="seals/Corridor-v0",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=100,
    kwargs=dict(
        length=100,
        reward_period=5,
    )
)

gym.register(
    id="seals/Corridor-v1",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=120,
    kwargs=dict(
        length=100,
        reward_period=10,
    )
)

gym.register(
    id="seals/Corridor-v2",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=120,
    kwargs=dict(
        length=100,
        reward_period=20,
    )
)

gym.register(
    id="seals/Corridor-v3",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=12,
    kwargs=dict(
        length=10,
        reward_period=None,
    )
)

gym.register(
    id="seals/Corridor-v4",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=120,
    kwargs=dict(
        length=100,
        reward_period=2,
    )
)

gym.register(
    id="seals/Corridor-v5",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=24,
    kwargs=dict(
        length=20,
        reward_period=2,
    )
)


gym.register(
    id="seals/Corridor-v6",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=120,
    kwargs=dict(
        length=100,
        reward_period=1,
    )
)

gym.register(
    id="seals/Corridor-v7",
    entry_point="derail.envs.corridor:CorridorEnv",
    max_episode_steps=200,
    kwargs=dict(
        length=100,
        reward_period=1,
    )
)

