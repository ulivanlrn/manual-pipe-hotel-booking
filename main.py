# import pandas as pd
from data.load_data import load_data
from preprocessing.feature_engineering import run_feature_engineering
from preprocessing.impute_data import run_imputation

# opening config
import yaml
with open("config/baseline_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# data loading
data = load_data(config["data"]["path"])

# initial cleaning
data = data[data['adults']>0]
data = data.drop(config["preprocessing"]["drop_from_start"], axis=1)
current_features = set(data.columns)

# imputing missing values
data = run_imputation(data, config, current_features)

# feature engineering
data = run_feature_engineering(data, config, current_features)

current_features = set(data.columns)
print(len(current_features))
print(current_features)