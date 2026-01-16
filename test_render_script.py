from train_utils import test_render_model
import time
import sys
import yaml

try:
    model_name = sys.argv[1]
except IndexError:
    print("Please provide the model name as a command-line argument.")
    sys.exit(1)



test_render_model(model_name)


