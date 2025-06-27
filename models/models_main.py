from data.load_data import load_data
from models.build_pipeline import build_pipeline
from models.evaluate import evaluate_model
from models.decision_threshold import tune_threshold
import yaml
from sklearn import set_config
import logging
import mlflow
from mlflow.models import infer_signature

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
key_params = pipeline.named_steps['model'].get_params()

# tuning decision threshold
if config["decision_threshold_tuning"]["flag"]:
    decision_threshold = config["decision_threshold_tuning"]["threshold"]
    pipeline = tune_threshold(pipeline, decision_threshold)
    key_params.update({'decision_threshold': decision_threshold})

# training and evaluating
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
metrics = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)

# logging experiment to MLFlow
mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_experiment("Experiment LogisticRegression")
signature = infer_signature(X_test, pipeline.predict(X_test)) # model signature

with mlflow.start_run():
    # log model hyperparameters and other key hyperparameters
    mlflow.log_params(key_params)
    # log model config
    mlflow.log_artifact(config_path)
    # log evaluation metrics
    mlflow.log_metrics(metrics)