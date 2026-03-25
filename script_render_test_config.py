from lib.rl_funcs import test_render_model
import time
import sys
import yaml


if len(sys.argv) >= 2:
    model_name = sys.argv[1]
    if "/" in model_name:
        dir , model_name = model_name.split("/")[-2], model_name.split("/")[-1]
    else:
        dir = ""
else:
    model_name = "Avoidance_relative_dynamic_5e5"
# MultiObsFrontierEnv_relative_DQN_5e5,16.78,0.50,01:10:19
# MultiObsFrontierEnv_distance_DQN_5e5,16.91,0.61,01:10:05
# MultiObsFrontierEnv_distance_DQN_dynamic_5e5,7.07,3.92,00:49:55
# MultiObsFrontierEnv_relative_DQN_dynamic_5e5,16.93,0.48,01:09:17
# MultiObsFrontierEnv_absolute_agent_DQN_5e5,15.92,1.80,00:48:53
# MultiObsFrontierEnv_absolute_agent_DQN_dynamic_5e5,16.46,1.52,00:58:35
    dir = "batchtry"

test_render_model(model_name, dir=dir)


