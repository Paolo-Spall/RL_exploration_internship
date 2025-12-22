# from gymnasium.envs.registration import register
import gymnasium as gym

# Register the environment so we can create it with gym.make()
gym.register(
    id="ExpGrid2D-v0",
    entry_point="exp_grid_2d_env:ExpGrid2D",
    max_episode_steps=300,  # Prevent infinite episodes
)