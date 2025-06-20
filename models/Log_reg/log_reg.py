from data.load_data import load_data
import yaml
from sklearn import set_config
import logging

set_config(transform_output='pandas')
logging.basicConfig(level=logging.DEBUG, filename='../../logs/models.log', filemode='w')

# opening config
config_name = "baseline"
config_path = f"../../config/models/{config_name}.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# data loading
preprocessing_config_name = "baseline"

X_train = load_data(f"../../data/X_train_{preprocessing_config_name}.csv")
X_test = load_data(f"../../data/X_test_{preprocessing_config_name}.csv")
y_train = load_data(f"../../data/y_train_{preprocessing_config_name}.csv")
y_test = load_data(f"../../data/y_test_{preprocessing_config_name}.csv")

logging.info("Loaded train, test and target data")