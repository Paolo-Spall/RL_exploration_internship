
from lib.train_utils import train_model, evaluate_model
import sys
import subprocess

def play_sound(file_path):
    # aplay is built into Ubuntu
    subprocess.run(["aplay", file_path])


if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = ['NewExplFrontStepEnv_rev-sort_DQN_1e5', 
                   'NewExplFrontStepEnv_DQN_1e5',
                   'NewExplFrontStepEnv_relative_DQN_1e5']
    # model_names = ['ExplFrontStepEnv_rev-sort_DQN_1e5', 
    #                'ExplFrontStepEnv_distances_DQN_1e5',
    #                'config_ExplFrontStepEnv_relative_DQN_1e5',]

for model_name in model_names:
    train_model(model_name)
    play_sound("/usr/share/sounds/sound-icons/start")
    evaluate_model(model_name)
    play_sound("/usr/share/sounds/sound-icons/finish")
    