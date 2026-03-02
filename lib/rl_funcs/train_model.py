
from lib.rl_funcs.learn_utils import training_time_monitor, get_policy_class, initialize_env,  wrap_model, my_checkenv, open_config



@training_time_monitor
def learn_model(model, total_timesteps):
    model.learn(total_timesteps=total_timesteps )
    return model

def train_model(model_name, check=False):
    config = open_config(model_name, save_copy=True)
    model_name = config['model_name']

    env = initialize_env(config)
    if check:
        my_checkenv(env, model_name)
    
    env = wrap_model(env, config)

    #env = TimeLimit(env, max_episode_steps=100)

    #   CREATE RL MODEL
    model_class = get_policy_class(config)
    model = model_class( env=env, **config['model']  )

    total_timesteps = config['training']['total_timesteps']
    
    print(f"\nTraining model: {model_name}")
    model, training_time = learn_model(model, total_timesteps)

    model.save("models/" + model_name)
    
    if config.get('wrapper'):
        if config['wrapper'].get('VecNormalize'):
            env.save(f"models/vec_normalize_{model_name}.pkl")
    env.close()

    output_file = f"models/evaluation_{model_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Training time: {training_time}\n")
    print(f"Evaluation results saved to {output_file}")
    return training_time


if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    train_model(config_file)