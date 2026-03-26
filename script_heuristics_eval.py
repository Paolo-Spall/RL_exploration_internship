#!/usr/bin/python3

from lib.rl_funcs import evaluate_heuristic
import sys
import os


if len(sys.argv) > 1:
    if sys.argv[1] == "-dir":
        # If the first argument is "-dir", read model names from the specified directory
        dir = sys.argv[2].strip("/")+"/" if len(sys.argv) > 2 else "configs/"
        in_dir = "configs/"+dir
        model_names = []
        for filename in os.listdir(in_dir):
            if filename.startswith("config_") and filename.endswith(".yaml"):
                model_name = filename[len("config_"):-len(".yaml")]
                model_names.append(model_name)
        out_dir = "models/"+dir.strip("/")
        os.makedirs(out_dir, exist_ok=True)
    else:
        dir = ""
        out_dir = "models/"+dir.strip("/")
        model_names = sys.argv[1:]
else:
    dir = ""
    out_dir = "models/"+dir.strip("/")
    model_names = ['MultiObsFrontierEnv_distance_DQN_dynamic_1e5']
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

os.makedirs(out_dir, exist_ok=True)

table_file = out_dir + "/evaluation_heuristics_results_0.txt"
n=0
while os.path.exists(table_file): 
    table_file = out_dir + f"/evaluation_heuristics_results_{n}.txt"
    n+=1

csv_file = table_file.replace(".txt", f"_{dir.strip('/')}.csv")

# open both table and csv files for writing and write the header to both
with open(table_file, "w") as f_table, open(csv_file, "w") as f_csv:
    header = "Model Name,Heuristic,Mean Reward,Std Reward\n"
    f_table.write(header)
    f_csv.write(header)

print()
print("========================================")

for model_name in model_names:
    for heuristic in ["distance", "info_gain"]:

        mean_reward, std_reward = evaluate_heuristic(model_name, 
                                                     n_episodes=30,
                                                     heuristic=heuristic,
                                                     dir=dir)
        
        with open(table_file, "a") as f_table, open(csv_file, "a") as f_csv:
            f_table.write(f"\n{model_name},{heuristic},{mean_reward:.2f},{std_reward:.2f}")
            f_csv.write(f"\n{model_name},{heuristic},{mean_reward:.2f},{std_reward:.2f}")
        
        print()

print("========================================")
print(f"All evaluations done. Results saved to {table_file}")
    

