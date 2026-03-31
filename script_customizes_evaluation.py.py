
from math import trunc

from lib.rl_funcs.custom_evaluation_model import custom_evaluation_model
from lib.rl_funcs.video_recording import record_model_video
import sys
import subprocess
import os


model_path = "models/batch_25-03_abs-agent-igain_not-sorted_padding-1_1e6/Avoidance_abs-ag-igain_static_not-sorted_pad-1_1e6/best_model_Avoidance_abs-ag-igain_static_not-sorted_pad-1_1e6/best_model.zip"
config_path = "models/batch_25-03_abs-agent-igain_not-sorted_padding-1_1e6/Avoidance_abs-ag-igain_static_not-sorted_pad-1_1e6/config_Avoidance_abs-ag-igain_static_not-sorted_pad-1_1e6.yaml"
save_dir = "models/batch_25-03_abs-agent-igain_not-sorted_padding-1_1e6/Avoidance_abs-ag-igain_static_not-sorted_pad-1_1e6/evaluation_best-model/"

n=1
while os.path.exists(save_dir): 
    n+=1
    save_dir = save_dir.rstrip('/') + f"_{n}/"

os.makedirs(save_dir, exist_ok=True)

model_filename = os.path.basename(model_path)

print()
no_video = input("If you DON'T want to record videos, type 'x': ").strip().lower() == 'x'

print()
print("========================================")  



mean_reward, std_reward = custom_evaluation_model(model_path, 
                                                     config_path, 
                                                     dir=save_dir)
    

if not no_video:
    record_model_video(
        model_name=model_path,
        output_path=f"{save_dir}video_{model_filename}.mp4",
        max_steps=None,
        fps=4,
        deterministic=True,
        check=False,
        seed=0,
        dir="",
        custom=True,
        config_path=config_path,
    )
else:
    print("Skipping video recording as per user request.")

print()
print("========================================")

