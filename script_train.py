
from lib.rl_funcs import train_model, evaluate_model
from lib.rl_funcs.video_recording import record_model_video
import sys
import subprocess
import os

def play_sound(file_path):
    # aplay is built into Ubuntu
    try:
        subprocess.run(["aplay", "-q", file_path], stdout=subprocess.DEVNULL)
    except Exception as e:
        pass  # If sound playback fails, just ignore it

if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = ['MultiObsFrontierEnv_absolute_DQN_1e5',]
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

table_file = "models/evaluation_results_0.txt"
n=0
while os.path.exists(table_file): 
    table_file = f"models/evaluation_results_{n}.txt"
    n+=1

with open(table_file, "w") as f:
    f.write("Model name,Mean reward,Std reward, Elapsed Time\n")

print()
no_video = input("If you DON'T want to record videos, type 'x': ").strip().lower() == 'x'

print()
print("========================================")  

for model_name in model_names:
    elapsed_time = train_model(model_name, check=True)
    play_sound("/usr/share/sounds/sound-icons/start")

    mean_reward, std_reward = evaluate_model(model_name)
    with open(table_file, "a") as f:
        f.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f},{elapsed_time}")
    
    if not no_video:
        record_model_video(
            model_name,
            output_path=f"models/video_{model_name}.mp4",
            max_steps=None,
            fps=4,
            deterministic=True,
            check=False,
            seed=42
        )
    else:
        print("Skipping video recording as per user request.")
    
    play_sound("/usr/share/sounds/sound-icons/glass-water-1.wav")
    print()
    print("========================================")

play_sound("/usr/share/sounds/sound-icons/finish")
print(f"All evaluations done. Results saved to {table_file}")
