import yaml
from sklearn import set_config
from sklearn.model_selection import train_test_split

from data.load_data import load_data
from preprocessing.feature_engineering import run_feature_engineering
from preprocessing.impute_data import Imputer

import logging

set_config(transform_output='pandas')
logging.basicConfig(level=logging.DEBUG, filename='../logs/preprocessing.log', filemode='w')

# opening config
config_name = "baseline"
config_path = f"../config/preprocessing/{config_name}.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# data loading
data = load_data(config["raw_data_path"])
logging.info("Raw data loaded")

# initial cleaning
data = data[data['adults']>0]
data = data.drop(config["drop_from_start"], axis=1)

# splitting data
X = data.drop(['is_canceled'], axis=1)
y = data['is_canceled']
test_size = config["train_test_split"]["test_size"]
random_state = config["train_test_split"]["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=y)

cat_features = list(X.select_dtypes('object').columns)
num_features = list(X.select_dtypes(exclude='object').columns)
all_features = set(X.columns)
feature_types = X.dtypes.value_counts()

logging.info("Cleaning and splitting into train and test sets complete")
logging.debug("{0} features before FE: {1}".format(len(all_features), all_features))
logging.debug(f"Feature types: {feature_types}")

# imputing missing values
imputer = Imputer(config, num_features, cat_features)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train_filled = (X_train.isna().sum().sum() == 0)
X_test_filled = (X_test.isna().sum().sum() == 0)
if X_train_filled & X_test_filled:
    logging.info("Imputation complete")
else:
    logging.warning("There are missing values after imputation")

# feature engineering
X_train = run_feature_engineering(X_train, config, all_features)
X_test = run_feature_engineering(X_test, config, all_features)

all_features = set(X_train.columns)
feature_types = X_train.dtypes.value_counts()

logging.info("Feature engineering complete")
logging.debug("{0} features after FE: {1}".format(len(all_features), all_features))
logging.debug(f"Feature types: {feature_types}")

# saving train and test data
X_train_path = f"../data/X_train_{config_name}.csv"
X_test_path = f"../data/X_test_{config_name}.csv"
y_train_path = f"../data/y_train_{config_name}.csv"
y_test_path = f"../data/y_test_{config_name}.csv"

X_train.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)

logging.info("Saving train and test data complete")