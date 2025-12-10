from stable_baselines3 import PPO
from exp_grid_2d_env import ExpGrid2D

env = ExpGrid2D(width=36, height=36, obstacle_prob=0.025, perc_range=2, render=False)

model = PPO("CnnPolicy", env, verbose=1, device="cpu")
# total_timesteps: it's.e.g., the number of actions the agent will take in the environment during training
model.learn(total_timesteps=10_000_000)
    
model.save("models/my_2d_grid_CNN_10e7_agent")
env.close()
