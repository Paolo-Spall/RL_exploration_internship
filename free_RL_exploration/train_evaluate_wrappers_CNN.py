from stable_baselines3 import DQN
from exp_grid_2d_env_multi_in import ExpGrid2D
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
import sys

train = True
if len(sys.argv) > 1 and sys.argv[1] == 'notrain':
    train = False


width=36
height=36
obstacle_prob=0.0
perc_range=5
max_steps=50
policy_type='MultiInputPolicy'
total_timesteps=100_000

model_name = 'my_2d_grid_DQN_MULTI_gpt-params_1e5_perc5'

## TRAINING
if train: 

    env = ExpGrid2D(width=width, 
                    height=height, 
                    obstacle_prob=obstacle_prob,
                    perc_range=perc_range, 
                    max_steps=max_steps, 
                    render_mode=None, 
                    policy_type=policy_type)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # If your env outputs HWC images, transpose to CHW for PyTorch
    env = VecTransposeImage(env)

    env = VecNormalize(env, norm_obs=False, norm_reward=True)#, clip_obs=10.)

    #env = TimeLimit(env, max_episode_steps=100)

    model = DQN(
        policy_type, env, device="cpu", verbose=1,
        learning_rate=5e-4,
        batch_size=64,
        buffer_size=50_000,
        exploration_fraction=0.7,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save("models/" + model_name)
    env.save(f"models/vec_normalize_{model_name}.pkl")
    env.close()

## EVALUATION

from stable_baselines3.common.evaluation import evaluate_policy

eval_env = ExpGrid2D(width=width, 
                     height=height, 
                     obstacle_prob=obstacle_prob,
                     perc_range=perc_range, 
                     max_steps=max_steps, 
                     render_mode=None, 
                    policy_type=policy_type)

eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# If your env outputs HWC images, transpose to CHW for PyTorch
eval_env = VecTransposeImage(eval_env)

eval_env = VecNormalize.load(f"models/vec_normalize_{model_name}.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False
# do not normalize observations at test time
eval_env.norm_obs = False

model = DQN.load(f"models/{model_name}", env=eval_env, device="cpu")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

output_file = f"models/evaluation_{model_name}.txt"
with open(output_file, "w") as f:
    f.write(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}\n")
print(f"Evaluation results saved to {output_file}")