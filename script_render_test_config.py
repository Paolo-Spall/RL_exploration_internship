from lib.rl_funcs import test_render_model
import time
import sys
import yaml


if len(sys.argv) >= 2:
    model_name = sys.argv[1]
    if "/" in model_name:
        dir , model_name = model_name.split("/")[-2], model_name.split("/")[-1]
else:
    model_name = "MultiObsFrontierEnv_relative_DQN_dynamic_1e5"
    # model_name = "MultiObsFrontierEnv_relative_DQN_1e5"
    # model_name = "MultiObsFrontierEnv_absolute_agent_DQN_dynamic_1e5"
    # model_name = "MultiObsFrontierEnv_distance_DQN_dynamic_1e5"
    dir = "batch_16-03_wrappers-static_nowrap-dynamic_reward"

test_render_model(model_name, dir=dir)


