
from math import log
import os

from lib.rl_funcs.learn_utils import training_time_monitor, get_policy_class, initialize_env,  wrap_model, my_checkenv, open_config
from stable_baselines3.common.callbacks import EvalCallback


@training_time_monitor
def learn_model(model, training_config, callback=None):
    # total_timesteps = training_config['total_timesteps']
    # log_interval = training_config.get('log_interval')
    if callback is not None:
        model.learn(**training_config, callback=callback)
    else:
        model.learn(**training_config)
    # if log_interval is not None:
    #     model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # else:
    #     model.learn(total_timesteps=total_timesteps )
    return model

def train_model(model_name, check=False, dir=""):
    exp_dir = f"models/{dir}{model_name}"
    #create folder for the experiment
    os.makedirs(exp_dir, exist_ok=True)
    
    config = open_config(model_name, save_copy=True, dir=dir)
    model_name = config['model_name']

    

    env = initialize_env(config, training=True)
    if check:
        if config.get('vectorize'):
            my_checkenv(env(), model_name)
        else:
            my_checkenv(env, model_name)
    
    env = wrap_model(env, config)

    #env = TimeLimit(env, max_episode_steps=100)

    #   CREATE RL MODEL
    config['model']['tensorboard_log'] = f"./{exp_dir}/tb_logs_{model_name}/"

    model_class = get_policy_class(config)
    model = model_class( env=env, **config['model']  )

    #total_timesteps = config['training']['total_timesteps']
    training_config = config['training']
    
    # Create separate evaluation environment
    eval_env = initialize_env(config, training=False)
    eval_env = wrap_model(eval_env, config)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.norm_obs = False
    
    # Setup EvalCallback
    eval_freq = config['training'].get('eval_freq', 20_000)
    n_eval_episodes = config['training'].get('n_eval_episodes', 20)
    best_model_path = f"{exp_dir}/best_model_{model_name}"
    eval_log_path = f"{exp_dir}/eval_logs_{model_name}"
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=eval_log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0
    )
    
    print(f"\nTraining model: {model_name}")
    print(f"Evaluation: every {eval_freq} steps, {n_eval_episodes} episodes per eval")
    model, training_time = learn_model(model, training_config, callback=eval_callback)

    model.save(f"{exp_dir}/{model_name}")
    
    if config.get('wrapper'):
        if config['wrapper'].get('VecNormalize'):
            env.save(f"{exp_dir}/vec_normalize_{model_name}.pkl")
    env.close()
    eval_env.close()

    output_file = f"{exp_dir}/evaluation_{model_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Training time: {training_time}\n")
        
    return training_time, exp_dir, best_model_path


if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    train_model(config_file)