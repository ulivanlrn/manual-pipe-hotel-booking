from data.load_data import load_data
from models.build_pipeline import build_pipeline
from models.train_evaluate import train_evaluate
import yaml
from sklearn import set_config
import logging

set_config(transform_output='pandas')
logging.basicConfig(level=logging.DEBUG, filename='../logs/models.log', filemode='w')

# MODEL AND CONFIG
model_type = "LogisticRegression"
config_name = "baseline"
config_path = f"../config/{model_type}/{config_name}.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logging.info(f"Experiment on: {model_type}")
logging.info(f"Config: {config_name}")

# DATA LOADING (data name suffix is the same as the name of the config for preprocessing,
# which was used to create a specific partition)
preprocessing_config_name = "baseline"

X_train = load_data(f"../data/X_train_{preprocessing_config_name}.csv")
X_test = load_data(f"../data/X_test_{preprocessing_config_name}.csv")
y_train = load_data(f"../data/y_train_{preprocessing_config_name}.csv")
y_test = load_data(f"../data/y_test_{preprocessing_config_name}.csv")

# converting target to 1d array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

logging.info(f"Data version: {preprocessing_config_name}")
logging.info("Data loading complete")

# building pipeline
pipeline = build_pipeline(config)

# training and evaluating the model
model_scores = train_evaluate(pipeline, X_train, y_train, X_test, y_test)
logging.info(f"Model scores in format [train_score, test_score]: {model_scores}")