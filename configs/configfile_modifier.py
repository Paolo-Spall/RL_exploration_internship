import yaml
import os

# list all the file in the current directory
file_list = os.listdir('.')

for file in file_list:
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        #config['training']['total_timesteps'] = 500000
        #config['model']['exploration_fraction'] = 0.7

        
        index = file.find('frac07')
        new_file_name = file[:index] + '_' + file[index:]
        # index = file.find('1e5')
        # new_file_name = file[:index] + '5e5' + file[index+3:]
        #new_file_name = file.strip('.yaml') + '_frac07' + '.yaml'
        os.rename(file, new_file_name)
        # write the updated config back to the file
        with open(new_file_name, 'w') as f:
            yaml.dump(config, f)