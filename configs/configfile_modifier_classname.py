import yaml
import os

# list all the file in the current directory
file_list = os.listdir('.')

for file in file_list:
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        # config['training']['total_timesteps'] = 200000
        # config['model']['exploration_fraction'] = 0.5
        config['env_class']= "MultiObsFrontAvoidanceEnv"
        
        mod_name = config['model_name']
        # # index = mod_name.find('1e5')
        # # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
        new_mod_name = mod_name.replace('MultiObsFrontierEnv', 'MultiObsFrontAvoidanceEnv')
        config['model_name'] = new_mod_name
        # new_mod_name = mod_name.replace('1e5', '2e5')

        # index = file.find('frac07')
        # new_file_name = file[:index] + '_' + file[index:]
        # index = file.find('1e5')
        # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
        # new_file_name = new_file_name.strip('.yaml') + '_frac02' + '.yaml'
        # new_file_name = file.strip('yaml')  + '.yaml'
        
        new_file_name = file.replace('MultiObsFrontierEnv', 'MultiObsFrontAvoidanceEnv')
        # new_file_name = "config_" + file + "_frac07.yaml"
        os.rename(file, new_file_name)


        # write the updated config back to the file
        with open(new_file_name, 'w') as f:
            yaml.dump(config, f)