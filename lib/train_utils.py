from stable_baselines3 import DQN, PPO
#from lib.free_RL_exploration.environments import ExpGrid2D, Simple2DGrid, Simple2DGridObs, Simple2DGridMultiObs
from lib.frontier_exploration.environments import ExplFrontStepEnv,\
                                                  ExplFrontStepDistancesEnv,\
                                                  NewExplFrontStepEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

import yaml



def train_model(model_name):
    config_file = f"configs/config_{model_name}.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)


    model_name = config['model_name']

    ## TRAINING
    env_class_str = config.get('env_class')
    env_class = globals()[env_class_str]


    env = env_class(**config['env'])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if config.get('wrapper').get('VecTransposeImage'):
    # If your env outputs HWC images, transpose to CHW for PyTorch
        env = VecTransposeImage(env)

    env = VecNormalize(env, **config['wrapper']['VecNormalize'])#, clip_obs=10.)

    #env = TimeLimit(env, max_episode_steps=100)

    class_str = config.get('model_class')
    model_class = globals()[class_str]

    model = model_class(
        env=env,
        verbose=1,
        **config['model']
    )

    model.learn(total_timesteps=config['training']['total_timesteps']   )


    model.save("models/" + model_name)
    env.save(f"models/vec_normalize_{model_name}.pkl")
    env.close()

    ## EVALUATION

def evaluate_model(model_name):
    config_file = f"configs/config_{model_name}.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)


    model_name = config['model_name']
    print(f"Evaluating model: {model_name}")

    env_class_str = config.get('env_class')
    env_class = globals()[env_class_str]


    eval_env = env_class(**config['env'])

    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    if config.get('wrapper').get('VecTransposeImage'):
    # If your env outputs HWC images, transpose to CHW for PyTorch
        eval_env = VecTransposeImage(eval_env)

    eval_env = VecNormalize.load(f"models/vec_normalize_{model_name}.pkl", eval_env)

    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False
    # do not normalize observations at test time
    eval_env.norm_obs = False

    class_str = config.get('model_class')
    model_class = globals()[class_str]

    model = model_class.load(f"models/{model_name}", env=eval_env, device="cpu")

    mean_reward, std_reward = evaluate_policy(model, eval_env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    output_file = f"models/evaluation_{model_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
    print(f"Evaluation results saved to {output_file}")

def test_render_model(model_name):
    config_file = f"configs/config_{model_name}.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['env']['render_mode'] = 'human'
    model_name = config['model_name']

    env_class_str = config.get('env_class')
    env_class = globals()[env_class_str]


    env = env_class(**config['env'])

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