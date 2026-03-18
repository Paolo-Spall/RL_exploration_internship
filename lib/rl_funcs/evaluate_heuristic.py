import os
import numpy as np


from lib.rl_funcs.learn_utils import get_policy_class, initialize_env,  wrap_model, my_checkenv, open_model_config


    ## EVALUATION

def run_greedy(env, max_steps, heuristic="distance"):
    tot_reward = 0
    obs, _ = env.reset()
    for i_step in range(max_steps):
        centroids = obs[::2]
        igain = obs[1::2]

        if heuristic == "distance":
            # select the closest centroid (the one with the smallest distance)
            mask = centroids == env.padding_value
            centroids[mask] = np.inf
            action = np.argmin(centroids)
        elif heuristic == "info_gain":
            # select the centroid with the highest information gain
            action = np.argmax(igain)
        
        #action = np.random.randint(0, len(centroids))
        obs, reward, term,  trunc, _ = env.step(action)
        tot_reward += reward
        if term or trunc:
            break

    return tot_reward

def evaluate_heuristic(model_name, n_episodes=30 ,check=False, heuristic="distance", render=False):
    config = open_model_config(model_name)
    config['obs_spec'] = {"ag_pos": False,
                          "i_gain": True,
                          "type": "distance"}
    if render:
        config['env']['render_mode'] = 'human'
    
    model_name = config['model_name']
    print(f"\nEvaluating heuristic: {heuristic}, on model: {model_name}")
    env = initialize_env(config)

    ## compute Gymnasium environment check
    if check:
        my_checkenv(env, model_name)

    max_steps = config.get('max_steps', 1000)
    ep_rewards = []
    
    for i_episode in range(n_episodes):
        tot_reward = run_greedy(env, max_steps, heuristic=heuristic)
        ep_rewards.append(tot_reward)

    mean_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    env.close()

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    if render:
        pass
    else:
        output_file = f"models/heuristic_results_{heuristic}_{model_name}.txt"
        with open(output_file, "a") as f:
            f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
        print(f"Evaluation results saved to {output_file}")

    return mean_reward, std_reward

if __name__ == "__main__":
    config_file = "configs/config_ex.yaml"
    