import os
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config, open_model_config, wrap_model_evaluation
from lib.rl_funcs.learn_utils import open_config_fromname
from lib.rl_funcs.evaluate_model import evaluate_action_stats

####   EVALUATION


def custom_evaluation_model(model_filepath, config_filepath, check=False, dir=""):
    if dir:
        dir = dir.strip('/') +'/'

    model_filename = os.path.basename(model_filepath).removesuffix(".zip")

    config_filename = os.path.basename(config_filepath)
    config = open_config_fromname(config_filepath)


    #model_name = config['model_name']
        
    print(f"\nEvaluating environment: {config_filename}")

    eval_env = initialize_env(config)

    ## compute Gymnasium environment check
    if check:
        my_checkenv(eval_env, config_filename)

    eval_env = wrap_model_evaluation(eval_env, config, 
                                     model_name=model_filename, 
                                     dir=dir)
    
    eval_env.training = False    # does not update them at test time
    eval_env.norm_reward = False # reward normalization is not needed at test time
    eval_env.norm_obs = False    # does not normalize observations at test time
    
    
    model_class = get_policy_class(config)

    model = model_class.load(f"{model_filepath}", env=eval_env, device="cpu")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

    # Evaluate action statistics
    action_stats = evaluate_action_stats(model, eval_env)

    eval_env.close()

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Action stats: mean = {action_stats['mean_action']:.4f} +/- {action_stats['std_action']:.4f}")


    output_file = f"{dir}evaluation_script_{model_filename}.txt"
    n=1
    while os.path.exists(output_file): 
        n+=1
        output_file = f"{dir}evaluation_script_{model_filename}_{n}.txt"
        

    with open(output_file, "a") as f:
        f.write(f"Evaluated model: {model_filename}\n")
        f.write(f"Config file: {config_filename}\n\n\n")
        f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
        f.write(f"Action statistics:\n")
        f.write(f"  Mean action: {action_stats['mean_action']:.4f}\n")
        f.write(f"  Std action: {action_stats['std_action']:.4f}\n")
        f.write(f"  Min action: {action_stats['min_action']:.4f}\n")
        f.write(f"  Max action: {action_stats['max_action']:.4f}\n")
        if 'action_distribution' in action_stats:
            f.write(f"  Action distribution: {action_stats['action_distribution']}\n")
        f.write(f"Episode statistics:\n")
        f.write(f"  Terminations: {action_stats['num_terminations']}/30 ({action_stats['termination_rate']:.2%})\n")
        f.write(f"  Truncations: {action_stats['num_truncations']}/30\n")
        f.write("\n")
    print(f"Evaluation results saved to {output_file}")

    return mean_reward, std_reward

if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    mean_reward, std_reward = custom_evaluation_model(config_file)