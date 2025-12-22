from stable_baselines3 import PPO, DQN
from exp_grid_2d_env import ExpGrid2D
import gymnasium as gym
#import free_RL_exploration  # Register custom environment

#from exp_grid_2d_env import ExpGrid2D

env = ExpGrid2D(   width=10, 
                   height=10, 
                   obstacle_prob=0.0, 
                   perc_range=2, 
                   render_mode=None, 
                   cnn=False)

model = DQN("MlpPolicy", env, verbose=1, device="cpu", learning_rate=1e-2, batch_size=16, buffer_size=10000)
# total_timesteps: it's.e.g., the number of actions the agent will take in the environment during training
model.learn(total_timesteps=100_000)

model.save("models/my_2d_grid_DQN_10e5_agent_batch16")
env.close()
