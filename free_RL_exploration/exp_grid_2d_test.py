from stable_baselines3 import PPO
from exp_grid_2d_env import ExpGrid2D
import time

width, height = 10, 10
obstacle_prob = 0.025
perc_range = 2
print("Creating environment...")
env = ExpGrid2D(width, height, obstacle_prob, perc_range=perc_range, render_mode="human", cnn=False)

print("Loading trained model...")
# Force CPU to avoid CUDA driver/runtime issues when loading the model
model = PPO.load("models/my_2d_grid_Mlp_10e5_agent", device="cpu")
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
