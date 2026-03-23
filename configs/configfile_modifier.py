import yaml
import os

# list all the file in the current directory
file_list = os.listdir('.')

countmod = 0

for file in file_list:
    if file.endswith('.yaml'):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        #config['training']['total_timesteps'] = 1000000
        config['model']['exploration_fraction'] = 0.5   

        mod_name = config['model_name']
        # # index = mod_name.find('1e5')
        # # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
        # new_mod_name = mod_name.replace('_frac07', '')
        
        #new_mod_name = mod_name.replace('5e5', '1e6')
        new_mod_name = mod_name.replace('frac02', 'frac05')
        # new_mod_name = mod_name + '_frac07'
        config['model_name'] = new_mod_name

        # index = file.find('frac07')
        # new_file_name = file[:index] + '_' + file[index:]
        # index = file.find('1e5')
        # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
        # new_file_name = new_file_name.strip('.yaml') + '_frac02' + '.yaml'
        # new_file_name = file.strip('yaml')  + '.yaml'
        
        #new_file_name = file.replace('5e5', '1e6')
        new_file_name = file.replace('frac02', 'frac05')
        # new_file_name = "config_" + file + "_frac07.yaml"
        os.rename(file, new_file_name)
        #new_file_name = file

        # write the updated config back to the file
        with open(new_file_name, 'w') as f:
            yaml.dump(config, f)
            countmod += 1
print(f"Number of files modified: {countmod}")