

import numpy as np

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_model_config




## RENDER MODEL WITHOUT SAVING VIDEO

def test_render_model(model_name, check=False, dir="", seed=None):
    """Render model in human mode without saving video"""
    if dir:
        dir = dir.strip("/")+"/"

    config = open_model_config(model_name, dir=dir)

    config['env']['render_mode'] = 'human'
    model_name = config['model_name']

    env = initialize_env(config)
    print(f"\nTesting rendering for model: {model_name}")

    if check:
        my_checkenv(env, model_name)

    model_class = get_policy_class(config)

    print("Loading trained model...")
    # Force CPU to avoid CUDA driver/runtime issues when loading the model
    model = model_class.load(f"models/{dir}{model_name}", env=env, device="cpu")
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