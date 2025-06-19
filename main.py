from sklearn import set_config
from sklearn.model_selection import train_test_split

from data.load_data import load_data
from preprocessing.feature_engineering import run_feature_engineering
from preprocessing.impute_data import Imputer

import logging

set_config(transform_output='pandas')
logging.basicConfig(level=logging.DEBUG, filename='main_logs.log', filemode='w')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123,
                                                    stratify=y)

cat_features = list(X.select_dtypes('object').columns)
num_features = list(X.select_dtypes(exclude='object').columns)
current_features = set(X.columns)

# imputing missing values
imputer = Imputer(config, num_features, cat_features)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# feature engineering
X_train = run_feature_engineering(X_train, config, current_features)
X_test = run_feature_engineering(X_test, config, current_features)

current_features = set(X_train.columns)
print(X_train.dtypes.value_counts())
print(len(current_features))
print(current_features)