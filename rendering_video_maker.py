"""Utilities to record and save a video of environment rendering."""

import numpy as np
import matplotlib

# Use a non-interactive backend to support headless video rendering
#matplotlib.use("Agg")

import imageio.v2 as imageio

from lib.rl_funcs.learn_utils import get_policy_class, initialize_env, my_checkenv, open_config
from lib.rendering_utils import fig_to_rgb

def _capture_frame(fig):
	fig.canvas.draw()
	width, height = fig.canvas.get_width_height()
	buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	return buffer.reshape((height, width, 3))


def record_model_render_video(
	model_name,
	output_path="video.mp4",
	max_steps=None,
	fps=10,
	deterministic=True,
	check=False,
):
	"""Record and save a video of the environment rendering for a trained model.

	Args:
		model_name: Name of the model (used to load config and weights).
		output_path: Path to the output video file (e.g., "video.mp4").
		max_steps: Maximum steps to record (defaults to env.max_steps).
		fps: Frames per second for the output video.
		deterministic: Whether to use deterministic actions.
		check: If True, run Gym environment checks.
	"""

	config = open_config(model_name)
	config["env"]["render_mode"] = None
	model_name = config["model_name"]

	env = initialize_env(config)
	print(f"\nRecording video for model: {model_name}")

	if check:
		my_checkenv(env, model_name)

	model_class = get_policy_class(config)

	print("Loading trained model...")
	model = model_class.load(f"models/{model_name}", env=env, device="cpu")
	print("Model loaded.")

	if not hasattr(env, "fig") or env.fig is None:
		if hasattr(env, "init_simulation_render"):
			env.init_simulation_render()

	fig = env.fig
	obs, _ = env.reset()
	

	#total_steps = max_steps if max_steps is not None else getattr(env, "max_steps", 250)
	total_steps = env.max_steps

	with imageio.get_writer(output_path, fps=fps) as writer:
		# Capture initial frame
		setattr(env, "render_mode", "rgb_array")
		writer.append_data(env.render())
		setattr(env, "render_mode", None)

		for _ in range(total_steps):
			action, _ = model.predict(obs, deterministic=deterministic)
			obs, reward, done, truncated, _ = env.step(action)

			# writer.append_data(_capture_frame(fig))
			setattr(env, "render_mode", "rgb_array")
			writer.append_data(env.render())
			setattr(env, "render_mode", None)

			if done or truncated:
				break

	env.close()
	print(f"Saved video to: {output_path}")

if __name__ == "__main__":
	import time
	model_name = "MultiObsFrontAvoidanceEnv_absolute_agent_DQN_1e5"
	itime = time.perf_counter()
	record_model_render_video(
		model_name,
		output_path="video.mp4",
		max_steps=None,
		fps=4,
		deterministic=True,
		check=False,
	)
	elapsed = time.perf_counter() - itime
	print(f"Elapsed time: {elapsed:.2f} seconds")