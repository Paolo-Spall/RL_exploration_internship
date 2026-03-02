#/usr/bin/python3

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_checker import check_env 

import yaml
import time
from time import perf_counter

from stable_baselines3 import DQN, PPO
from lib.free_RL_exploration.environments import SimpleTargetAgentEnv,\
                                                    SimpleTargetAgentFlatEnv, \
                                                    MultiobsSimpleTargetAgentEnv
from lib.frontier_exploration.environments import MultiObsFrontierEnv, \
                                                  MultiObsFrontAvoidanceEnv


# decorator to monitor function time
def training_time_monitor(func):
    def wrapper(*args, **kwargs):
        print()
        timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
        itime = perf_counter()
        print(f"--- Starting at {timestr} ---")

        model = func(*args, **kwargs)

        elapsed = perf_counter() - itime
        timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
        print(f"--- Finishing at {timestr} ---")
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Total Execution Time: {formatted_time} (HH:MM:SS)")
        print()
        return model, formatted_time
    return wrapper

def checkenv_script(model_name):
    config = open_config(model_name)
    env = initialize_env(config)
    my_checkenv(env, model_name)
    

def my_checkenv(env, model_name):
    print(f"\nChecking environment for model: {model_name}")
    #print(f"Enironment class: {config.get('env_class')}")
    check_env(env, warn=True)
    print("Environment check done.")


def open_config(model_name, save_copy=True):
    config_file = f"configs/config_{model_name}.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    if save_copy:
        yaml.dump(config, open(f"models/config_{model_name}.yaml", 'w'))  # Save a copy of the config in the models folder
    return config


def get_policy_class(config):
    class_str = config.get('model_class')
    model_class = globals()[class_str]
    return model_class

def initialize_env(config):
    env_class_str = config.get('env_class')
    env_class = globals()[env_class_str]

    env = env_class(**config['env'])
    
    return env

def wrap_model(env, config, model_name=None, evaluation=False):
    if config.get('wrapper') == None:
        return env
    wrapper_config = config.get('wrapper')

    if wrapper_config.get('Monitor'):
        env = Monitor(env)
    if wrapper_config.get('DummyVecEnv'):
        env = DummyVecEnv([lambda: env])

    if wrapper_config.get('VecTransposeImage'):
    # If your env outputs HWC images, transpose to CHW for PyTorch
        env = VecTransposeImage(env)
    
    if wrapper_config.get('VecNormalize'):
        if evaluation:
            env = VecNormalize.load(f"models/vec_normalize_{model_name}.pkl", env)
        else:
            env = VecNormalize(env, **config['wrapper']['VecNormalize'])#, clip_obs=10.)
    return env
