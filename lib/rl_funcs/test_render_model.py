

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config




## RENDER MODEL WITHOUT SAVING VIDEO

def test_render_model(model_name, check=False):
    """Render model in human mode without saving video"""
    config = open_config(model_name)

    config['env']['render_mode'] = 'human'
    model_name = config['model_name']

    env = initialize_env(config)
    print(f"\nTesting rendering for model: {model_name}")

    if check:
        my_checkenv(env, model_name)

    model_class = get_policy_class(config)

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
    pass