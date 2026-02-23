#!/usrbin/python3
import yaml
import pprint
import os

filename = "config_new.yaml"
filedir = "configs/"

with open(filedir + filename, 'r') as file:
    config = yaml.safe_load(file)

model_name_list = []

for obs_type in [ 'absolute', 'relative','distance']:
    for info_gain in [True, False]:
        for ag_pos in [False, True]:
            config['env_class'] = "MultiObsFrontierEnv"
            config['env']['render_mode'] = None
            config['env']['target_discovery_percent'] = 0.9
            config['env']['obs_spec'] = {'type': obs_type,
                                      'ag_pos': ag_pos,
                                      'i_gain': info_gain}
            model_name = "MultiObsFrontierEnv_"
            model_name += f"{obs_type}_"
            if ag_pos:
                model_name += "agent_"
            if info_gain:
                model_name += "iGain_"
            model_name += "DQN_1e5"
            config['model_name'] = model_name
            if ag_pos:
                config['model']['policy'] = 'MultiInputPolicy'
            else:
                config['model']['policy'] = 'MlpPolicy'
            config['model']['verbose'] = 0
            config['training']['total_timesteps'] = 100000
            out_filename = filedir + "config_" + model_name + ".yaml"
            with open(out_filename, 'w') as outfile:
                yaml.dump(config, outfile)
            model_name_list.append(model_name)
print("Created config files for models:")
pprint.pprint(model_name_list)

script = "#!/usr/bin/bash\n"
for model_name in model_name_list:
    script += f"python3 script_train.py {model_name}\n"

script_file = "run_all_models_0.sh"
n=0
while os.path.exists(script_file): 
    script_file = f"run_all_models_{n}.sh"
    n+=1
with open(filedir + script_file, 'w') as outfile:
    outfile.write(script)
