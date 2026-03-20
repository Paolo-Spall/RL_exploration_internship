
from math import log

from lib.rl_funcs.learn_utils import training_time_monitor, get_policy_class, initialize_env,  wrap_model, my_checkenv, open_config



@training_time_monitor
def learn_model(model, training_config):
    # total_timesteps = training_config['total_timesteps']
    # log_interval = training_config.get('log_interval')
    model.learn(**training_config)
    # if log_interval is not None:
    #     model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # else:
    #     model.learn(total_timesteps=total_timesteps )
    return model

def train_model(model_name, check=False, dir=""):
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
    config['model']['tensorboard_log'] = f"./models/{dir}tb_logs_{model_name}/"

    model_class = get_policy_class(config)
    model = model_class( env=env, **config['model']  )

    #total_timesteps = config['training']['total_timesteps']
    training_config = config['training']
    
    print(f"\nTraining model: {model_name}")
    model, training_time = learn_model(model, training_config)

    model.save(f"models/{dir}{model_name}")
    
    if config.get('wrapper'):
        if config['wrapper'].get('VecNormalize'):
            env.save(f"models/{dir}vec_normalize_{model_name}.pkl")
    env.close()

    output_file = f"models/{dir}evaluation_{model_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Training time: {training_time}\n")
        
    return training_time


if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    train_model(config_file)