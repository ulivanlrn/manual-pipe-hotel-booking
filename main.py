# import pandas as pd
from data.load_data import load_data
from preprocessing.impute_data import fill_in_constant
from preprocessing.impute_data import random_impute

# opening config
import yaml
with open("config/baseline_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# data loading
data = load_data(config["data"]["path"])

# cleaning data
data = data.drop(config["preprocessing"]["drop_from_start"], axis=1)
data = data[data['adults']>0]
data['children'] = fill_in_constant(data['children'],
                                    config["preprocessing"]["children_impute_value"])
random_impute(data, 'country')
random_impute(data, 'agent')
