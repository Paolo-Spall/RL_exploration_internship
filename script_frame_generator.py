#!/usr/bin/python3
from fileinput import filename
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib



if __name__ == "__main__":
    import sys
    sys.path.append(".")
from lib.frontier_exploration.planning.planning_utils import is_in_grid, is_obstacle, render_path
from lib.utils import find_agent
from lib.grid_env.obst_grid_agent_env import ObstGridAgentEnv
from lib.frontier_exploration.environments import MultiObsFrontierEnv

if __name__ == "__main__":
    width, height = 20, 20
    obstacle_prob = 0.035
    target_discovery_percent = 0.9
    perc_range = 3
    
    truncations = 0
    terminations = 0
    obs_type ='relative'# 'absolute', 'distance']:#['relative', 'absolute', 'distance']:
    info_gain =False#, False]:# [True, False]
    ag_pos = False#, False ]:# [True, False]
    # input("Press Enter to create the new environment...")
    #print(f"Obs type: {obs_type}, Info gain: {info_gain}, Agent pos: {ag_pos}")
    print("Creating environment...")
    env = MultiObsFrontierEnv(  width=width, 
                                height=height, 
                                obstacle_prob=obstacle_prob, 
                                target_discovery_percent=target_discovery_percent,
                                perc_range=perc_range, 
                                max_front_cluster_size=20,
                                render_mode=  "human", # "human", None,
                                sorting=True,
                                reverse=False,
                                obs_spec={'type':obs_type,
                                        'ag_pos':ag_pos,
                                        'i_gain':info_gain},
                                static_obstacles=True,
                                static_obstacles_seed=44
                                )
    env.reset( init_agent_pos=(11,11))
    #env.agent_pos = np.array((13,8))
    cont = 0

    savecont = 0

    

    while cont<50:
        env.step(0)
        save = input('save?')
        if save == 'y':
            while os.path.exists(f'files/frontiers_grid_{savecont}.joblib'):
                savecont += 1
            filename = f'files/frontiers_grid_{savecont}.joblib'
            joblib.dump([env.obs_grid,
                        env.clusters,
                        env.centroids],
                    filename)
            print(f"Saved to {filename}")
            savecont +=1
        cont+=1
# import joblib 
# env.init_simulation_render()
# save = input('Save?')
# if save == 'y':
#     joblib.dump([env.obs_grid,
#                 env.clusters,
#                 env.centroids],
#              'frontiers_grid.joblib')