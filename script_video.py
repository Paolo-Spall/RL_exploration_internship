from lib.rl_funcs.learn_utils import get_policy_class, initialize_env,  wrap_model, my_checkenv, open_config
# from lib.rl_funcs.test_render_model import one_step_frame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

def one_step_frame(frame, ax, obs, env, model):
    # if frame == 0:
    #     print("if true")
    #     obs, _ = env.reset()
    # else:
        # print("else true")
    action, _ = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, _ = env.step(action)
    ax = [env.ax_env, env.ax_obs]
    if done or truncated:
        return None
    return ax, obs, env, model

model_name = "MultiObsFrontAvoidanceEnv_relative_iGain_DQN_1e5"
model_name = "MultiObsFrontierEnv_relative_iGain_DQN_1e5"
check = True

config = open_config(model_name)

config['env']['render_mode'] = 'human'
model_name = config['model_name']

env = initialize_env(config)
print(f"\nTesting rendering for model: {model_name}")

# if check:
#     my_checkenv(env, model_name)

model_class = get_policy_class(config)

print("Loading trained model...")
# Force CPU to avoid CUDA driver/runtime issues when loading the model
model = model_class.load(f"models/{model_name}", env=env, device="cpu")
print("Model loaded.")

fig = env.fig
ax1 = env.ax_env
ax2 = env.ax_obs
ax = [ax1, ax2]
obs, _ = env.reset()

anim = FuncAnimation(fig, partial(one_step_frame, ax=ax, obs=obs, env=env, model=model), #scatter=scat), 
                            frames=env.max_steps, 
                            blit=False,
                            repeat=False )
anim.save("video.mp4")
# plt.show()