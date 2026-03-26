
from math import exp

from lib.rl_funcs import train_model, evaluate_model
from lib.rl_funcs.video_recording import record_model_video
import sys
import subprocess
import os

def play_sound(file_path):
    # aplay is built into Ubuntu
    try:
        subprocess.run(["aplay", "-q", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        pass  # If sound playback fails, just ignore it

if len(sys.argv) > 1:
    if sys.argv[1] == "-dir":
        # If the first argument is "-dir", read model names from the specified directory
        dir = sys.argv[2].strip("/")+"/" #if len(sys.argv) > 2 else "configs/"
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
    model_names = ['MultiObsFrontierEnv_absolute_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
 'MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_relative_DQN_1e5',
 'MultiObsFrontierEnv_relative_agent_DQN_1e5',
 'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
 'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_distance_DQN_1e5',
 'MultiObsFrontierEnv_distance_agent_DQN_1e5',
'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
 'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5',]
#  'MultiObsFrontAvoidanceEnv_absolute_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_agent_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_agent_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_agent_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_agent_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_agent_iGain_DQN_1e5']
#         

#     
 

table_file = out_dir + "/evaluation_results_0.txt"
n=0
while os.path.exists(table_file): 
    table_file = out_dir + f"/evaluation_results_{n}.txt"
    n+=1

csv_file = table_file.replace(".txt", f"_{dir.strip('/')}.csv")

# open both table and csv files for writing and write the header to both
with open(table_file, "w") as f_table, open(csv_file, "w") as f_csv:
    header = "Model Name,Mean Reward,Std Reward,Mean Action,Std Action,Elapsed Time\n"
    f_table.write(header)
    f_csv.write(header)

print()
#no_video = input("If you DON'T want to record videos, type 'x': ").strip().lower() == 'x'

#setting by default recording of the video active
no_video = False

print()
print("========================================")  

for model_name in model_names:
    elapsed_time , experiment_dir = train_model(model_name ,check=True, dir=dir)
    play_sound("/usr/share/sounds/sound-icons/start")

    mean_reward, std_reward, mean_action, std_action = evaluate_model(model_name, dir= experiment_dir)
    with open(table_file, "a") as f_table, open(csv_file, "a") as f_csv:
        f_table.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f},{mean_action:.4f},{std_action:.4f},{elapsed_time}")
        f_csv.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f},{mean_action:.4f},{std_action:.4f},{elapsed_time}")

    if not no_video:
        record_model_video(
            model_name,
            output_path=f"{experiment_dir}/video_{model_name}.mp4",
            max_steps=None,
            fps=4,
            deterministic=True,
            check=False,
            seed=0,
            dir=experiment_dir,
        )
    else:
        print("Skipping video recording as per user request.")
    
    play_sound("/usr/share/sounds/sound-icons/glass-water-1.wav")
    print()
    print("========================================")

play_sound("/usr/share/sounds/sound-icons/finish")
print(f"All evaluations done. Results saved to {table_file}")
