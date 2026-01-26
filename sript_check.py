#!/usr/bin/python3

from lib.train_utils import checkenv_script

import sys


if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = ['NewExplFrontStepEnv_distance_DQN_1e5']
        # 'NewExplFrontStepEnv_rev-sort_DQN_1e5', 
        #            'NewExplFrontStepEnv_DQN_1e5',
        #            'NewExplFrontStepEnv_relative_DQN_1e5']
    # model_names = ['ExplFrontStepEnv_rev-sort_DQN_1e5', 
    #                'ExplFrontStepEnv_distances_DQN_1e5',
    #                'config_ExplFrontStepEnv_relative_DQN_1e5',]

for model_name in model_names:
    checkenv_script(model_name)