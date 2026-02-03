from stable_baselines3 import DQN, PPO
#from lib.free_RL_exploration.environments import ExpGrid2D, Simple2DGrid, Simple2DGridObs, Simple2DGridMultiObs
from lib.frontier_exploration.environments import MultiObsFrontierEnv, \
                                                  MultiObsFrontAvoidanceEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env 

import time
from datetime import timedelta
from time import perf_counter

import yaml

def checkenv_script(model_name):
    config = open_config(model_name)
    env = initialize_model(config)
    my_checkenv(env,config, model_name)
    

def my_checkenv(env,config, model_name):
    print(f"\nChecking environment for model: {model_name}")
    #print(f"Enironment class: {config.get('env_class')}")
    check_env(env, warn=True)
    print("Environment check done.")


def open_config(model_name):
    config_file = f"configs/config_{model_name}.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_model(config):

    ## TRAINING
    env_class_str = config.get('env_class')
    env_class = globals()[env_class_str]

    env = env_class(**config['env'])
    
    return env

def wrap_model(env, config):
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if config.get('wrapper').get('VecTransposeImage'):
    # If your env outputs HWC images, transpose to CHW for PyTorch
        env = VecTransposeImage(env)

    env = VecNormalize(env, **config['wrapper']['VecNormalize'])#, clip_obs=10.)
    return env



def training_time_monitor(func):
    def wrapper(*args, **kwargs):
        timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
        itime = perf_counter()
        print(f"--- Starting at {timestr} ---")

        model = func(*args, **kwargs)

        elapsed = perf_counter() - itime
        timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
        print(f"--- Finishing at {timestr} ---")
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Total Execution Time: {formatted_time} (HH:MM:SS)")
        
        return model, formatted_time
    return wrapper

@training_time_monitor
def learn_model(model, total_timesteps):
    model.learn(total_timesteps=total_timesteps )
    return model

def train_model(model_name, check=False):
    config = open_config(model_name)
    model_name = config['model_name']

    env = initialize_model(config)
    if check:
        my_checkenv(env,config, model_name)
    env = wrap_model(env, config)

    #env = TimeLimit(env, max_episode_steps=100)

    #   CREATE RL MODEL
    class_str = config.get('model_class')
    model_class = globals()[class_str]
    model = model_class( env=env, **config['model']  )
    total_timesteps = config['training']['total_timesteps']
    
    print(f"\nTraining model: {model_name}")
    model, training_time = learn_model(model, total_timesteps)

    # timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
    # itime = perf_counter()
    # print(f"\nTraining model: {model_name}")
    # print(f"--- Starting at {timestr} ---")

    # model.learn(total_timesteps=config['training']['total_timesteps'] )

    # elapsed = perf_counter() - itime
    # timestr = time.strftime('%H:%M:%S %Y-%m-%d', time.localtime())
    # print(f"--- Finishing at {timestr} ---")
    # formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    # print(f"Total Execution Time: {formatted_time} (HH:MM:SS)")

    model.save("models/" + model_name)
    env.save(f"models/vec_normalize_{model_name}.pkl")
    env.close()

    output_file = f"models/evaluation_{model_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Training time: {training_time}\n")
    print(f"Evaluation results saved to {output_file}")
    return training_time

    ## EVALUATION

def evaluate_model(model_name, check=False):
    config = open_config(model_name)

    model_name = config['model_name']
    print(f"\nEvaluating model: {model_name}")

    eval_env = initialize_model(config)

    ## compute Gymnasium environment check
    if check:
        my_checkenv(eval_env,config, model_name)

    eval_env = wrap_model(eval_env, config)

    
    eval_env.training = False    # does not update them at test time
    eval_env.norm_reward = False # reward normalization is not needed at test time
    eval_env.norm_obs = False    # does not normalize observations at test time
    
    class_str = config.get('model_class')
    model_class = globals()[class_str]

    model = model_class.load(f"models/{model_name}", env=eval_env, device="cpu")

    mean_reward, std_reward = evaluate_policy(model, eval_env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    output_file = f"models/evaluation_{model_name}.txt"
    with open(output_file, "a") as f:
        f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
    print(f"Evaluation results saved to {output_file}")
    return mean_reward, std_reward

def test_render_model(model_name, check=False):
    config = open_config(model_name)

    config['env']['render_mode'] = 'human'
    model_name = config['model_name']

    env = initialize_model(config)
    print(f"\nTesting rendering for model: {model_name}")

    if check:
        my_checkenv(env,config, model_name)

    class_str = config.get('model_class')
    model_class = globals()[class_str]

    print("Loading trained model...")
    # Force CPU to avoid CUDA driver/runtime issues when loading the model
    model = model_class.load(f"models/{model_name}", env=env, device="cpu")
    print("Model loaded.")

    obs, _ = env.reset()

    print("Starting exploration...")
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, _ = env.step(action)
        print(f"Step: {step}, Reward: {reward}, Done: {done}")
        print(obs)

        #time.sleep(0.1)

        if done:
            print("Exploration complete!")
            break
        if truncated:
            print("Exploration truncated!")
            break

    env.close()

if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    train_model(config_file)
    evaluate_model(config_file)