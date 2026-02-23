#!/usr/bin/python3

from lib.rl_funcs import evaluate_heuristic
import sys
import os


if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = [
 #'MultiObsFrontAvoidanceEnv_absolute_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_absolute_agent_DQN_1e5',
 #'MultiObsFrontAvoidanceEnv_absolute_iGain_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_absolute_agent_iGain_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_relative_DQN_1e5',
 #'MultiObsFrontAvoidanceEnv_relative_agent_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_relative_iGain_DQN_1e5',
 #'MultiObsFrontAvoidanceEnv_relative_agent_iGain_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_distance_DQN_1e5',
 #'MultiObsFrontAvoidanceEnv_distance_agent_DQN_1e5',
 'MultiObsFrontAvoidanceEnv_distance_iGain_DQN_1e5',]
 #'MultiObsFrontAvoidanceEnv_distance_agent_iGain_DQN_1e5']
    # model_name = 'MultiObsFrontierEnv_distance_iGain_DQN_1e5'
    #'MultiObsFrontierEnv_absolute_DQN_1e5'
#  'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_DQN_1e5',
#  'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_DQN_1e5',
# 'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5']
 

# table_file = "models/evaluation_results_0.txt"
# n=0
# while os.path.exists(table_file): 
#     table_file = f"models/evaluation_results_{n}.txt"
#     n+=1

# with open(table_file, "w") as f:
#     f.write("Model name,Mean reward,Std reward, Elapsed Time\n")

# print()
# no_video = input("If you DON'T want to record videos, type 'x': ").strip().lower() == 'x'

print()

for heuristic in ["distance", "info_gain"]:
    input(f"Press Enter to render heuristic: {heuristic}...")
    mean_reward, std_reward = evaluate_heuristic(model_name, 
                                                 n_episodes=1,
                                                 heuristic=heuristic,
                                                 render=True)
    
    print()
    

