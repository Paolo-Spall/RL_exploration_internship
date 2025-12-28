from stable_baselines3 import PPO, DQN
from environments import Simple2DGrid


print("Creating environment...")
size = 10
env = Simple2DGrid(size=size, render_mode=True)

model_name = "my_2d_grid_MultiInput_1e6_agent"
model_path = f"models/{model_name}"

model = DQN("MultiInputPolicy", env, verbose=1, device="cpu",
    learning_rate=0.0005,
  batch_size=64,
  buffer_size=50000,
  exploration_fraction=0.7,
  exploration_final_eps=0.05,
  target_update_interval=500,
  train_freq=4)



print("Loading trained model...")
# Force CPU to avoid CUDA driver/runtime issues when loading the model
model = DQN.load(model_path, device="cpu")
print("Model loaded.")

obs, _ = env.reset()

print("Starting exploration...")
for step in range(60):
    action, _ = model.predict(obs, deterministic=True)
    print("Action taken:", action, "action type:", type(action))
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step: {step}, Reward: {reward}")

    #time.sleep(0.1)

    if terminated:
        print("Exploration complete!")
        break
    if truncated:
        print("Max steps reached, ending exploration.")
        break

env.close()
