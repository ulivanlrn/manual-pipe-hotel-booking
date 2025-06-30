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
model_version = "baseline"
config_path = f"../config/{model_type}/{model_version}.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logging.info(f"Experiment on: {model_type}")
logging.info(f"Config: {model_version}")

# DATA LOADING (data name suffix is the same as the name of the config for preprocessing,
# which was used to create a specific partition)
data_version = "baseline"

X_train = load_data(f"../data/X_train_{data_version}.csv")
X_test = load_data(f"../data/X_test_{data_version}.csv")
y_train = load_data(f"../data/y_train_{data_version}.csv")
y_test = load_data(f"../data/y_test_{data_version}.csv")

# converting target to 1d array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

logging.info(f"Data version: {data_version}")
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
mlflow.set_experiment(f"Experiment {model_type}")
signature = infer_signature(X_test, pipeline.predict(X_test)) # model signature

with mlflow.start_run():
    # log model hyperparameters and other key hyperparameters
    mlflow.log_params(key_params)
    # log model config
    mlflow.log_artifact(config_path)
    # log evaluation metrics
    mlflow.log_metrics(metrics)
    # set tags of the run
    mlflow.set_tag("model_version", model_version)
    mlflow.set_tag("data_version", data_version)
    # log the model
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        signature=signature,
        input_example=X_test[:1]
    )
