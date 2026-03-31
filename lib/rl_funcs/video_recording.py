
from csv import writer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import imageio.v2 as imageio
import os
import yaml

from gymnasium.wrappers import RecordVideo

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config, open_model_config



def record_model_video(
    model_name,
    output_path="video.mp4",
    max_steps=None,
    fps=10,
    deterministic=True,
    check=False,
    seed=None,
    dir="",
    notrunc_flag=False,
    custom = None,
    config_path = None,):
    """Record and save a video of the environment rendering for a trained model.

    Args:
        model_name: Name of the model (used to load config and weights).
        output_path: Path to the output video file (e.g., "video.mp4").
        max_steps: Maximum steps to record (defaults to env.max_steps).
        fps: Frames per second for the output video.
        deterministic: Whether to use deterministic actions.
        check: If True, run Gym environment checks.
    """

    if dir:
        dir = dir.strip('/') +'/'
    
    if custom is not None:
        config_path = config_path.replace(".yaml", "")
        config_file = f"{config_path}.yaml"
        
        model_path = model_name
        model_name = os.path.basename(model_path)
    else:
        config_file = f"{dir}config_{model_name}.yaml"
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)



    config["env"]["render_mode"] = None
    
    if notrunc_flag:
        notrunc_str = "no-trunc_"
        config['env']['padding_truncation'] = False
    else:
        notrunc_str = ""
        
    model_name = config["model_name"]

    env = initialize_env(config)
    print(f"\nRecording video for model: {model_name}")

    if check:
        my_checkenv(env, model_name)

    model_class = get_policy_class(config)

    print("Loading trained model...")
    if custom:
        model = model_class.load(model_path, env=env, device="cpu")
    else:
        model = model_class.load(f"{dir}{model_name}", env=env, device="cpu")
    print("Model loaded.")

    if not hasattr(env, "fig") or env.fig is None:
        if hasattr(env, "init_simulation_render"):
            env.init_simulation_render()

    fig = env.fig
    obs, _ = env.reset(seed=seed)


    #total_steps = max_steps if max_steps is not None else getattr(env, "max_steps", 250)
    total_steps = env.max_steps

    with imageio.get_writer(output_path, fps=fps) as writer:
        # Capture initial frame
        setattr(env, "render_mode", "rgb_array")
        writer.append_data(env.render())
        setattr(env, "render_mode", None)

        for _ in range(total_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            intaction = int(action)
            obs, reward, done, truncated, _ = env.step(intaction)

            # writer.append_data(_capture_frame(fig))
            setattr(env, "render_mode", "rgb_array")
            writer.append_data(env.render())
            setattr(env, "render_mode", None)

            if done or truncated:
                break

    env.close()
    print(f"Saved video to: {output_path}")

## RENDER MODEL AND SAVE VIDEO USING VIDEOWRAPPER

def record_model_video_wrapper(model_name, check=False, seed=None):
    """Render model and save video using VideoWrapper"""
    config = open_config(model_name)

    config['env']['render_mode'] = 'rgb_array'
    model_name = config['model_name']

    env = initialize_env(config)
    env = RecordVideo(env, video_folder="models", episode_trigger=lambda x: x == 0)
    print(f"\nRecording video for model: {model_name}")

    if check:
        my_checkenv(env, model_name)

    model_class = get_policy_class(config)

    print("Loading trained model...")
    # Force CPU to avoid CUDA driver/runtime issues when loading the model
    model = model_class.load(f"models/{model_name}", env=env, device="cpu")
    print("Model loaded.")

    obs, _ = env.reset(seed=seed)

    print("Starting exploration...")
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, _ = env.step(action)
        #print(f"Step: {step}, Reward: {reward}, Done: {done}")

        #time.sleep(0.1)

        if done:
            print("Exploration complete!")
            break
        if truncated:
            print("Exploration truncated!")
            break

    env.close()
    old_name = "models/rl-video-episode-0.mp4"
    # new_name = f"models/video_{model_name}.mp4"
    new_name = f"video_{model_name}.mp4"

    if os.path.exists(old_name):
        os.rename(old_name, new_name)


## FuncAnimation based function for recording video

def make_video(model_name, one_step_frame, check=False, ):
    """FuncAnimation based function for recording video"""
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

    fig = env.fig
    ax1 = env.ax_env
    ax2 = env.ax_obs
    ax = [ax1, ax2]

    anim = FuncAnimation(fig, partial(one_step_frame, ax, env, model), #scatter=scat), 
                             frames=env.max_steps, 
                             blit=False,
                            repeat=False )
    plt.show()

def one_step_frame(frame, ax, env, model):
    """Animate-like funtion"""
    if frame == 0:
        obs, _ = env.reset()
    else:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, _ = env.step(action)
    ax = [env.ax_env, env.ax_obs]
    return ax