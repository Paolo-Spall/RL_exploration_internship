import yaml
import os

# list all the file in the current directory
file_list = os.listdir('.')

for file in file_list:
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training']['total_timesteps'] = 500000

        

        # write the updated config back to the file
        with open(file, 'w') as f:
            yaml.dump(config, f)