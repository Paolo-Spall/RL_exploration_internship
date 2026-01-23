from stable_baselines3 import PPO, DQN
from lib.frontier_exploration.environments import ExplFrontGymStepCentr


model_name = "ExplFrontGymStepCentr_DQN_1e5_Monitor"
model_path = "models/" + model_name

print("Creating environment...")
width=20
height=20 
obstacle_prob = 0.05
perc_range = 3
target_discovery_percent=0.9
max_steps_per_episode = 500

env = ExplFrontGymStepCentr(width=width, 
                             height=height, 
                             obstacle_prob = obstacle_prob,
                            perc_range = perc_range,
                              target_discovery_percent=target_discovery_percent,
                             max_steps = max_steps_per_episode,
                             render_mode='human')



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
