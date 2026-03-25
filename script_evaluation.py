
from math import trunc

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

    if "--no-trunc" in sys.argv:
        sys.argv.remove("--no-trunc")
        notrunc_str = "no-trunc_"
        notrunc_flag = True
    else:
        notrunc_str = ""
        notrunc_flag = False

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
        model_names = sys.argv[1:]

else:
    model_names = ['MultiObsFrontierEnv_absolute_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
 'MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_relative_DQN_1e5',
 'MultiObsFrontierEnv_relative_agent_DQN_1e5',]
#  'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_DQN_1e5',
# 'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5']
 
#     model_names = [
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


table_file = 'models/' + dir + f"evaluation_script_{notrunc_str}results_0.txt"
n=0
while os.path.exists(table_file): 
    table_file = 'models/' + dir + f"evaluation_script_{notrunc_str}results_{n}.txt"
    n+=1

csv_file = table_file.replace(".txt", f"_{dir.strip('/')}.csv")


with open(table_file, "w") as f:
    f.write("Model name,Mean reward,Std reward,Mean action,Std action\n")

print()
no_video = input("If you DON'T want to record videos, type 'x': ").strip().lower() == 'x'

print()
print("========================================")  

for model_name in model_names:

    mean_reward, std_reward, mean_action, std_action = evaluate_model(model_name, notrunc_flag=notrunc_flag)
    with open(table_file, "a") as f, open(csv_file, "w") as f_csv:
        f.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f},{mean_action:.4f},{std_action:.4f}")
        f_csv.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f},{mean_action:.4f},{std_action:.4f}")
    

    if not no_video:
        record_model_video(
            model_name,
            output_path=f"models/{dir}video_{model_name}_{notrunc_str}.mp4",
            max_steps=None,
            fps=4,
            deterministic=True,
            check=False,
            seed=0,
            dir=dir,
            notrunc_flag=notrunc_flag,
        )
    else:
        print("Skipping video recording as per user request.")
    
    print()
    print("========================================")

play_sound("/usr/share/sounds/sound-icons/finish")
print(f"All evaluations done. Results saved to {table_file}")
