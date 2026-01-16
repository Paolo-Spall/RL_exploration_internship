#!/usr/bin/python3
from frontier_exploration.environments import ExplFrontGymStepCentrSort
from frontier_exploration.utils import greedy_index

width, height = 20, 20
obstacle_prob = 0.05
target_discovery_percent = 0.9
perc_range = 3
print("Creating environment...")
env = ExplFrontGymStepCentrSort(width=width, 
                    height=height, 
                    obstacle_prob=obstacle_prob, 
                    target_discovery_percent=target_discovery_percent,
                    perc_range=perc_range, 
                    render_mode="human", 
                #    render_mode=False, 
                    policy_type='mlp')

# check_env(env, warn=True)
# print("Env checked.")
# exit()
print("Resetting environment...")
obs, _ = env.reset()
print("Stepping through the environment...")

term = False

while not term:
    centroids = obs['frontier_centroids'].reshape(-1,2)
    agent_pos = obs['agent_position']
    action = greedy_index(centroids, agent_pos)
    obs, reward, term,  trunc, _ = env.step(action)