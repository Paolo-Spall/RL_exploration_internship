

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_model_config




## RENDER MODEL WITHOUT SAVING VIDEO

def test_render_model(model_name, check=False, dir=""):
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

    obs, _ = env.reset()
    tot_reward = 0

    print("Starting exploration...")
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        intaction = int(action)
        
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
    env.close()



if __name__ == "__main__":
    pass