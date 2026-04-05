

import numpy as np
import os

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config_fromname, open_model_config




## RENDER MODEL WITHOUT SAVING VIDEO

def test_render_model(model_filepath, config_filepath, check=False, dir="", seed=None):
    """Render model in human mode without saving video"""
    if dir:
        dir = dir.strip("/")+"/"

    model_filename = os.path.basename(model_filepath).removesuffix(".zip")

    config_filename = os.path.basename(config_filepath)
    config = open_config_fromname(config_filepath)

    config['env']['render_mode'] = 'human'
    model_name = config['model_name']

    env = initialize_env(config)
    print(f"\nTesting rendering for model: {model_filename}")

    if check:
        my_checkenv(env, model_name)

    model_class = get_policy_class(config)

    print("Loading trained model...")
    # Force CPU to avoid CUDA driver/runtime issues when loading the model
    model = model_class.load(model_filepath, env=env, device="cpu")
    print("Model loaded.")

    obs, _ = env.reset(seed=seed)
    tot_reward = 0
    all_actions = []

    print("Starting exploration...")
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        intaction = int(action)
        all_actions.append(intaction)
        
        obs, reward, done, truncated, _ = env.step(intaction)
        print(f"Step: {step}, Reward: {reward}, Done: {done}")
        tot_reward += reward

        #time.sleep(0.1)

        if done:
            print("Exploration complete!")
            break
        if truncated:
            print("Exploration truncated!")
            break

    print(f"Total reward: {tot_reward}")
    
    # Compute and print action statistics
    all_actions = np.array(all_actions)
    mean_action = float(np.mean(all_actions))
    std_action = float(np.std(all_actions))
    print(f"Action statistics: mean = {mean_action:.4f}, std = {std_action:.4f}")
    
    # Check if all actions are 0 or (if reverse) all equal to 9
    all_zero = np.all(all_actions == 0)
    all_nine = np.all(all_actions == 9)
    is_reverse = config['env'].get('reverse', True)
    
    if is_reverse and all_nine:
        print("⚠ All actions were 9 (reverse=True)!")
    elif not is_reverse and all_zero:
        print("⚠ All actions were 0 (reverse=False)!")
    else:
        print("✓ Actions varied.")
    
    env.close()



if __name__ == "__main__":
    pass