import yaml
import os

dumped = 0

# list all the file in the current directory
dir_list = os.listdir('.')
for directory in dir_list:
    if os.path.isdir(directory):
        # rename the directori adding "_reverse to the end of the name
        

        file_list = os.listdir(directory)
        for file_rel in file_list:
            file = os.path.join(directory, file_rel)
            if file.endswith('.yaml'):
                with open(file, 'r') as f:
                    config = yaml.safe_load(f)
                
                #config['training']['total_timesteps'] = 1000000
                #config['model']['exploration_fraction'] = 0.5
                # config['env']['reverse'] = True
                config['training']['log_interval'] = 100
                # delete the item with key 'reverse' of dict config:
                # if 'reverse' in config:
                #     del config['reverse'] 

                # mod_name = config['model_name']
                # # index = mod_name.find('1e5')
                # # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
                # new_mod_name = mod_name.replace('_frac07', '')
                
                #new_mod_name = mod_name.replace('5e5', '1e6')
                # new_mod_name = mod_name + '_reverse'
                #config['model_name'] = new_mod_name

                # index = file.find('frac07')
                # new_file_name = file[:index] + '_' + file[index:]
                # index = file.find('1e5')
                # new_file_name = file[:index] + 'frac02_' + '2e5' + file[index+3:]
                # new_file_name = new_file_name.strip('.yaml') + '_frac02' + '.yaml'
                # new_file_name = file.strip('yaml')  + '.yaml'
                
                #new_file_name = file.replace('5e5', '1e6')
                #new_file_name = file.strip('.yaml') + '_reverse' + '.yaml'
                #os.rename(file, new_file_name)

                new_file_name = file
                # write the updated config back to the file
                with open(new_file_name, 'w') as f:
                    yaml.dump(config, f)
                    dumped += 1
        
        #new_directory = directory + '_reverse'
        
        #os.rename(directory, new_directory)
print(f'{dumped} files modified')