
from lib.train_utils import train_model, evaluate_model
import sys

if len(sys.argv) > 1:
    model_names = sys.argv[1:]
else:
    model_names = ['simple_2D_obs']

for model_name in model_names:
    train_model(model_name)
    evaluate_model(model_name)