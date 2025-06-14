# import pandas as pd
from sklearn.model_selection import train_test_split
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

# splitting data
X = data.drop(['is_canceled'], axis=1)
y = data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,
                                                    stratify=y)

cat_features = X.select_dtypes('object').columns
num_features = X.select_dtypes(exclude='object').columns

current_features = set(X.columns)

# imputing missing values
data = run_imputation(data, config, current_features)

# feature engineering
data = run_feature_engineering(data, config, current_features)

current_features = set(data.columns)
# print(len(current_features))
# print(current_features)