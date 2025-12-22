#!/usr/bin/python3
# -*- coding: utf-8 -*-
from stable_baselines3 import DQN
from exp_grid_2d_env import ExpGrid2D
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy

eval_env = ExpGrid2D(width=10, 
                     height=10, 
                     obstacle_prob=0.0,
                     perc_range=2, 
                     max_steps=50, 
                     render_mode=None, 
                     cnn=False)

eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize.load("models/vec_normalize_2d_grid_DQN_CNN_gpt-params_10k.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

model = DQN.load("models/my_2d_grid_DQN_CNN_gpt-params_10k", env=eval_env, device="cpu")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")