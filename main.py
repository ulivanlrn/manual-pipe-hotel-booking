# import pandas as pd
from data.load_data import load_data
from preprocessing.impute_data import fill_in_constant
from preprocessing.impute_data import random_impute
from preprocessing.feature_engineering import stays_func
from preprocessing.feature_engineering import room_type
from preprocessing.feature_engineering import deposit_type
from utils import set1_in_set2

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
feature_flags = config["features"]["flags"]
# a variable that collects all old features which contributed to new ones
# and are not needed afterward
drop_after_fe = set()

if feature_flags["total_nights"] &\
    set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
    data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    drop_after_fe = drop_after_fe.union({'stays_in_weekend_nights', 'stays_in_week_nights'})

if feature_flags["stays_format"] &\
    set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
    data['stays_format'] = data.apply(stays_func, axis=1)

if feature_flags["total_guests"] &\
    set1_in_set2({'adults', 'children', 'babies'}, current_features):
    data['total_guests'] = data['adults'] + data['children'] + data['babies']
    drop_after_fe = drop_after_fe.union({'adults', 'children', 'babies'})

if feature_flags["room_assigned_equal_reserved"] &\
    set1_in_set2({'assigned_room_type', 'reserved_room_type'}, current_features):
    data['room_assigned_equal_reserved'] = data.apply(room_type, axis=1)
    drop_after_fe = drop_after_fe.union({'assigned_room_type', 'reserved_room_type'})

if feature_flags["map_deposit_type"]:
    data['deposit_type'] = data.apply(deposit_type, axis=1)

data = data.drop(drop_after_fe, axis=1)

current_features = set(data.columns)