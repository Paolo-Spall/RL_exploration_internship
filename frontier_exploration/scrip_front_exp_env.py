#!/usr/bin/python3
from environments.env_planning_frontiers import FrontierExplorationEnv
from utils import closest


width, height =20, 30
obstacle_prob = 0.1
perc_range = 5
max_steps = 250

print("Creating environment...")
env = FrontierExplorationEnv(width, 
                    height, 
                    obstacle_prob, 
                    perc_range=perc_range,
                    max_steps=max_steps,    
                    render_mode="human", 
                    policy_type='mlp')

# check_env(env, warn=True)
print("Resetting environment...")
obs, _ = env.reset()
print("Stepping through the environment...")
terminated = False
truncated = False
while not (terminated or truncated):
    action = closest(obs['frontier_centroids'], obs['agent_position'])
    obs, reward, terminated, truncated, info = env.step(action)
print("Done.")
print("Terminated:", terminated, "Truncated:", truncated)