from stable_baselines3 import DQN
from exp_grid_2d_env import ExpGrid2D
from gymnasium.wrappers import TimeLimit

env = ExpGrid2D(width=10, height=10, obstacle_prob=0.0,
                perc_range=2, max_steps=50, render_mode=None, cnn=False)

#env = TimeLimit(env, max_episode_steps=100)

model = DQN(
    "MlpPolicy", env, device="cpu", verbose=1,
    learning_rate=5e-4,
    batch_size=64,
    buffer_size=50_000,
    exploration_fraction=0.7,
    exploration_final_eps=0.05,
    target_update_interval=500,
    train_freq=4,
)

model.learn(total_timesteps=200_000)

model.save("models/my_2d_grid_DQN_gpt-params_fr-freq4_200k")
env.close()
