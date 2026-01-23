#!/usr/bin/python3
# from lib.frontier_exploration.environments import ExplFrontGymStepCentrSort
# from lib.utils import greedy_index
from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv
import matplotlib.pyplot as plt

width, height = 20, 20
obstacle_prob = 0.05
print("Creating environment...")
env = ObstGridAgentEnv(width=width, 
                    height=height, 
                    render_mode="human", 
                    obstacle_prob=obstacle_prob
                )

print("Resetting environment...")
env.reset(init_agent_pos=(1,5))
print("Rendering the environment...")
env.render()
plt.show()

for action in range(4):
    print("Taking action: ", action)
    env.step(action)
    plt.show()