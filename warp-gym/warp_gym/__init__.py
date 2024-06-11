from gymnasium.envs.registration import register

# register(
#     id="warp_gym/CartPole-v0",
#     entry_point="warp_gym.envs:CartPoleEnv",
#     max_episode_steps=500,
# )
register(
    id="warp_gym/InvertedPendulum-v0",
    entry_point="warp_gym.envs:InvertedPendulumEnv",
    max_episode_steps=1000,
    kwargs={"fps": 60, "substeps": 10, "urdf_path": None},
)
