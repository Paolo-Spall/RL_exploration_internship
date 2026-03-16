import yaml
import os

# list all the file in the current directory
file_list = os.listdir('.')

for file in file_list:
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training']['total_timesteps'] = 500000
        # config['model']['exploration_fraction'] = 0.2

        
        index = file.find('1e5')
        new_file_name = file[:index] + '5e5' + file[index+4:]
        os.rename(file, new_file_name)
        # write the updated config back to the file
        with open(new_file_name, 'w') as f:
            yaml.dump(config, f)