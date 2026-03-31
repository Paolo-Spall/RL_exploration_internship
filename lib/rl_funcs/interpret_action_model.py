import os
import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append('.')   

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config, open_model_config, wrap_model_evaluation
from lib.rl_funcs.learn_utils import open_config_fromname

####   EVALUATION

def interpret_action_stats(model, env, n_eval_episodes=30):
    """
    Evaluate and return statistics about actions selected by the model.
    
    Args:
        model: The trained model
        env: The evaluation environment
        n_eval_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Statistics including mean, std, min, max of actions, action distribution,
              episode termination/truncation counts, and reward statistics
    """
    all_actions = []
    all_actions_sorted = []
    all_rewards = []
    num_terminations = 0
    num_truncations = 0
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            centroids = env.unwrapped.centroids
            agent_pos = env.unwrapped.agent_pos
            centroids_ordermap = np.argsort( np.linalg.norm(centroids - agent_pos, axis=1) )
            action, _ = model.predict(obs, deterministic=True)
            all_actions_sorted.append(centroids_ordermap[action])
            all_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            all_rewards.append(reward)
            done = terminated or truncated
        
        # Track termination vs truncation
        if terminated:
            num_terminations += 1
        elif truncated:
            num_truncations += 1
    
    all_actions = np.array(all_actions)
    all_actions_sorted = np.array(all_actions_sorted)
    all_rewards = np.array(all_rewards)
    action_stats = {
        'mean_action': float(np.mean(all_actions)),
        'std_action': float(np.std(all_actions)),
        'min_action': float(np.min(all_actions)),
        'max_action': float(np.max(all_actions)),
        'mean_action_sorted': float(np.mean(all_actions_sorted)),
        'std_action_sorted': float(np.std(all_actions_sorted)),
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'num_terminations': num_terminations,
        'num_truncations': num_truncations,
        'termination_rate': float(num_terminations / n_eval_episodes),
    }
    
    # If discrete action space, add distribution
    if len(all_actions.shape) == 1:  # Discrete actions
        unique, counts = np.unique(all_actions, return_counts=True)
        unique_sorted, counts_sorted = np.unique(all_actions_sorted, return_counts=True)
        total_actions = len(all_actions)
        total_actions_sorted = len(all_actions_sorted)
        action_stats['action_distribution'] = {int(a): float(c / total_actions) for a, c in zip(unique, counts)}
        action_stats['action_distribution_sorted'] = {int(a): float(c / total_actions_sorted) for a, c in zip(unique_sorted, counts_sorted)}
    
    return action_stats

def interpret_model(model_filepath, config_filepath, check=False, dir=""):
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

    #mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)

    # Evaluate action statistics
    action_stats = interpret_action_stats(model, eval_env)

    eval_env.close()

    #print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Action stats: mean = {action_stats['mean_action']:.4f} +/- {action_stats['std_action']:.4f}")

    output_file = f"{dir}INTERPRETATION_{model_filename}.txt"
    with open(output_file, "a") as f:
        f.write(f"Reward statistics:\n")
        f.write(f"  Mean reward: {action_stats['mean_reward']:.4f}\n")
        f.write(f"  Std reward: {action_stats['std_reward']:.4f}\n")
        f.write("\n")
        f.write(f"Action statistics:\n")
        f.write(f"  Mean action: {action_stats['mean_action']:.4f}\n")
        f.write(f"  Std action: {action_stats['std_action']:.4f}\n")
        f.write(f"  Min action: {action_stats['min_action']:.4f}\n")
        f.write(f"  Max action: {action_stats['max_action']:.4f}\n")
        if 'action_distribution' in action_stats:
            f.write(f"  Action distribution: {action_stats['action_distribution']}\n")
        
        f.write("\n")
        f.write(f"Sorted action statistics:\n")
        f.write(f"  Mean action (sorted): {action_stats['mean_action_sorted']:.4f}\n")
        f.write(f"  Std action (sorted): {action_stats['std_action_sorted']:.4f}\n")
        f.write(f"  Action distribution (sorted): {action_stats.get('action_distribution_sorted', 'N/A')}\n")
        f.write("\n")
        f.write(f"Episode statistics:\n")
        f.write(f"  Terminations: {action_stats['num_terminations']}/30 ({action_stats['termination_rate']:.2%})\n")
        f.write(f"  Truncations: {action_stats['num_truncations']}/30\n")
    print(f"Evaluation results saved to {output_file}")

    return 0

if __name__ == "__main__":
    config_path = "models/batch_25-03_abs-agent_not-sorted_padding-1_1e6/config_Avoidance_abs-ag_static_not-sorted_pad-1_1e6.yaml"
    model_path = "models/batch_25-03_abs-agent_not-sorted_padding-1_1e6/Avoidance_abs-ag_static_not-sorted_pad-1_1e6.zip"
    interpret_model(model_path, config_path, 
                    check=True, 
                    dir="models/batch_25-03_abs-agent_not-sorted_padding-1_1e6")