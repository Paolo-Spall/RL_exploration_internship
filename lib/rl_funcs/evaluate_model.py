import os
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config, open_model_config, wrap_model_evaluation


####   EVALUATION

def evaluate_action_stats(model, env, n_eval_episodes=10):
    """
    Evaluate and return statistics about actions selected by the model.
    
    Args:
        model: The trained model
        env: The evaluation environment
        n_eval_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Statistics including mean, std, min, max of actions and action distribution
    """
    all_actions = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    all_actions = np.array(all_actions)
    
    action_stats = {
        'mean_action': float(np.mean(all_actions)),
        'std_action': float(np.std(all_actions)),
        'min_action': float(np.min(all_actions)),
        'max_action': float(np.max(all_actions)),
    }
    
    # If discrete action space, add distribution
    if len(all_actions.shape) == 1:  # Discrete actions
        unique, counts = np.unique(all_actions, return_counts=True)
        action_stats['action_distribution'] = {int(a): int(c) for a, c in zip(unique, counts)}
    
    return action_stats

def evaluate_model(model_name, check=False, dir="", notrunc_flag=False):
    config = open_model_config(model_name, dir=dir)
    if notrunc_flag:
        notrunc_str = "no-trunc_"
        print("Evaluation with NO TRUNCATION of episodes.")
        config['env']['padding_truncation'] = False
    else:
        notrunc_str = ""

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

    # Evaluate action statistics
    action_stats = evaluate_action_stats(model, eval_env)

    eval_env.close()

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Action stats: mean = {action_stats['mean_action']:.4f} +/- {action_stats['std_action']:.4f}")

    output_file = f"models/{dir}evaluation_{notrunc_str}{model_name}.txt"
    with open(output_file, "a") as f:
        f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
        f.write(f"Action statistics:\n")
        f.write(f"  Mean action: {action_stats['mean_action']:.4f}\n")
        f.write(f"  Std action: {action_stats['std_action']:.4f}\n")
        f.write(f"  Min action: {action_stats['min_action']:.4f}\n")
        f.write(f"  Max action: {action_stats['max_action']:.4f}\n")
        if 'action_distribution' in action_stats:
            f.write(f"  Action distribution: {action_stats['action_distribution']}\n")
        f.write("\n")
    print(f"Evaluation results saved to {output_file}")

    return mean_reward, std_reward, action_stats['mean_action'], action_stats['std_action']

if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    mean_reward, std_reward, mean_action, std_action = evaluate_model(config_file)