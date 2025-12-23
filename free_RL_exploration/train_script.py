
from train_utils import train_model, evaluate_model
import yaml

config_files = ['configs/config_exp3.yaml']

for config_file in config_files:
    #train_model(config_file)
    evaluate_model(config_file)