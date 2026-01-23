from stable_baselines3 import DQN, PPO
from lib.frontier_exploration.environments import ExplFrontGymStepCentrSort
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage

model_name = "ExplFrontGymStepCentrSort_DQN_1e5"
path = "models/"
model_path = path + model_name

width=20
height=20 
obstacle_prob = 0.05
perc_range = 3
target_discovery_percent=0.9
max_steps_per_episode = 500

env = ExplFrontGymStepCentrSort(width=width, 
                             height=height, 
                             obstacle_prob = obstacle_prob,
                            perc_range = perc_range,
                              target_discovery_percent=target_discovery_percent,
                              max_steps=max_steps_per_episode,
                             render_mode=False)

env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = DQN("MultiInputPolicy", env, verbose=1, device="cpu",
    learning_rate=0.0005,
  batch_size=64,
  buffer_size=50000,
  exploration_fraction=0.7,
  exploration_final_eps=0.05,
  target_update_interval=500,
  train_freq=4)

# total_timesteps: it's.e.g., the number of actions the agent will take in the environment during training
model.learn(total_timesteps=100_000)
    
model.save(model_path)

vec_norm_path = path + f"vec_normalize_{model_name}.pkl"

env.save(vec_norm_path)
env.close()

eval_env = ExplFrontGymStepCentrSort(width=width, 
                                  height=height, 
                                  obstacle_prob=obstacle_prob,
                                  perc_range=perc_range,
                                  target_discovery_percent=target_discovery_percent,
                                  render_mode=False)

eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize.load(vec_norm_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
eval_env.norm_obs = False

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

output_path = f"models/evaluation_{model_name}.txt"

with open(output_path, "w") as f:
    f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
print(f"Evaluation results saved to {output_path}")
eval_env.close()
