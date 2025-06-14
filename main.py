# import pandas as pd
from data.load_data import load_data
from preprocessing.impute_data import fill_in_constant
from preprocessing.impute_data import random_impute
from preprocessing.feature_engineering import run_feature_engineering

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
if 'children' in current_features:
    data['children'] = fill_in_constant(data['children'],
                                        config["preprocessing"]["children_impute_value"])
for feature in ['country', 'agent']:
    if feature in current_features:
        random_impute(data, feature)

# feature engineering
data = run_feature_engineering(data, config, current_features)

current_features = set(data.columns)
print(len(current_features))
print(current_features)