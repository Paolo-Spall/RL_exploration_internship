import os

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config, wrap_model_evaluation


####   EVALUATION

def evaluate_model(model_name, check=False, dir=""):
    config = open_config(model_name, dir=dir)

    model_name = config['model_name']
        
    print(f"\nEvaluating model: {model_name}")

    eval_env = initialize_env(config)

    ## compute Gymnasium environment check
    if check:
        my_checkenv(eval_env, model_name)

    eval_env = wrap_model_evaluation(eval_env, config, 
                                     model_name=model_name, 
                                     dir=dir)
    
    eval_env.training = False    # does not update them at test time
    eval_env.norm_reward = False # reward normalization is not needed at test time
    eval_env.norm_obs = False    # does not normalize observations at test time
    
    
    model_class = get_policy_class(config)

    model = model_class.load(f"models/{dir}{model_name}", env=eval_env, device="cpu")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)

    eval_env.close()

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    output_file = f"models/{dir}evaluation_{model_name}.txt"
    with open(output_file, "a") as f:
        f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
    print(f"Evaluation results saved to {output_file}")

    return mean_reward, std_reward

if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    evaluate_model(config_file)