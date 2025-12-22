from stable_baselines3 import PPO, DQN
from exp_grid_2d_env_multi_in import ExpGrid2D
import time

width, height = 36, 36
obstacle_prob = 0.0
perc_range = 5
print("Creating environment...")
env = ExpGrid2D(width, 
                height, 
                obstacle_prob, 
                perc_range=perc_range,
                max_steps=50, 
                render_mode="human", 
                policy_type='MultiInputPolicy')

print("Loading trained model...")
# Force CPU to avoid CUDA driver/runtime issues when loading the model
model = DQN.load("models/my_2d_grid_DQN_MULTI_gpt-params_1e4", device="cpu")
print("Model loaded.")

obs, _ = env.reset()

print("Starting exploration...")
for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    print("Action taken:", action, "action type:", type(action))
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {step}, Reward: {reward}, Done: {done}")

    #time.sleep(0.1)

    if done:
        print("Exploration complete!")
        break

env.close()
