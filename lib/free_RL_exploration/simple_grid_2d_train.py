from stable_baselines3 import DQN, PPO
from environments import Simple2DGrid
from stable_baselines3.common.evaluation import evaluate_policy

size = 10
env = Simple2DGrid(size=size, render_mode=False)

model_name = "my_2d_grid_MultiInput_1e6_agent"

model = DQN("MultiInputPolicy", env, verbose=1, device="cpu",
    learning_rate=0.0005,
  batch_size=64,
  buffer_size=50000,
  exploration_fraction=0.7,
  exploration_final_eps=0.05,
  target_update_interval=500,
  train_freq=4)

# total_timesteps: it's.e.g., the number of actions the agent will take in the environment during training
model.learn(total_timesteps=1_000_000)
    
model.save(model_name)

eval_env = Simple2DGrid(size=size, render_mode=False)

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

output_file = f"evaluation_{model_name}.txt"

with open(output_file, "w") as f:
    f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
print(f"Evaluation results saved to {output_file}")

env.close()
