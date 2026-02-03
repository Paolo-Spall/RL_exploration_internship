import sys

from lib.rl_funcs.learn_utils import training_time_monitor
from lib.rl_funcs.video_record import record_model_render_video, test_render_model_videowrapper


if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
     model_names = ['MultiObsFrontAvoidanceEnv_absolute_agent_DQN_1e5']
#      ['MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
#  'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_DQN_1e5',
#  'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_DQN_1e5']

# model_names = ['MultiObsFrontAvoidanceEnv_absolute_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_agent_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_absolute_agent_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_relative_agent_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_agent_iGain_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_DQN_1e5',
#  'MultiObsFrontAvoidanceEnv_distance_agent_DQN_1e5']
    
#    model_names =['MultiObsFrontierEnv_absolute_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_DQN_1e5',
#  'MultiObsFrontierEnv_absolute_agent_DQN_1e5',
#  'MultiObsFrontierEnv_relative_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_relative_DQN_1e5',
#  'MultiObsFrontierEnv_relative_agent_DQN_1e5',
#  'MultiObsFrontierEnv_distance_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_iGain_DQN_1e5',
#  'MultiObsFrontierEnv_distance_DQN_1e5',
#  'MultiObsFrontierEnv_distance_agent_DQN_1e5']


for model_name in model_names:
    wrapped = training_time_monitor(test_render_model_videowrapper)
    wrapped(model_name, check=False)
    
    
	
    training_time_monitor(record_model_render_video)(
		model_name,
		output_path="video.mp4",
		max_steps=None,
		fps=4,
		deterministic=True,
		check=False,
	)
    