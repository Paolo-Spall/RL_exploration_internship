
from lib.train_utils import train_model, evaluate_model
import sys
import subprocess
import os

def play_sound(file_path):
    # aplay is built into Ubuntu
    subprocess.run(["aplay", "-q", file_path], stdout=subprocess.DEVNULL)


if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = ['MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_absolute_DQN_1e5',
 'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
 'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
 'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_relative_DQN_1e5',
 'MultiObsFrontierEnv_relative_agent_DQN_1e5',
 'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
 'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5',
 'MultiObsFrontierEnv_distance_DQN_1e5',
 'MultiObsFrontierEnv_distance_agent_DQN_1e5']

table_file = "models/evaluation_results_0.txt"
n=0
while os.path.exists(table_file): 
    table_file = f"models/evaluation_results_{n}.txt"
    n+=1

with open(table_file, "w") as f:
    f.write("Model name,Mean reward,Std reward")
    

for model_name in model_names:
    train_model(model_name, check=True)
    play_sound("/usr/share/sounds/sound-icons/start")
    mean_reward, std_reward = evaluate_model(model_name)
    with open(table_file, "a") as f:
        f.write(f"\n{model_name},{mean_reward:.2f},{std_reward:.2f}")
    print()
    print("========================================")
    print()
play_sound("/usr/share/sounds/sound-icons/finish")
print(f"All evaluations done. Results saved to {table_file}")
