from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def fill_in_constant(column, value):
    return column.fillna(value)

def random_impute(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isna()].index
    df.loc[df[feature].isna(), feature] = random_sample

def run_imputation(data, config, current_features):
    if 'children' in current_features:
        data['children'] = fill_in_constant(data['children'],
                                            config["preprocessing"]["children_impute_value"])
    # NOTE: directly depends on the sample
    for feature in ['country', 'agent']:
        if feature in current_features:
            random_impute(data, feature)

    return data

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, config, num_features, cat_features):
        self.config = config
        self.num_features = num_features
        self.cat_features = cat_features
        self.knn_neighbors = config["preprocessing"]["knn_neighbors"]
        self.knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.column_transformer = ColumnTransformer(
            transformers=[

            ]
        )
    def fit(self, x):
        self.knn_imputer.fit(x)
    def transform(self, x):
        pass
