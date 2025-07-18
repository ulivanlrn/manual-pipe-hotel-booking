from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import List

class Imputer(BaseEstimator, TransformerMixin):
    """
    A custom general imputer, which applies KNNImputer to numerical features and
    SimpleImputer to categorical features. Column 'children' is handled separately.

    Attributes:
        config (dict): Configuration dictionary.
        num_features (list): List of numerical features.
        cat_features (list): List of categorical features.
    """
    def __init__(self, config: dict, num_features: List[str], cat_features: List[str]):
        self.config = config
        self.num_features = num_features
        self.cat_features = cat_features
        self.knn_neighbors = config["knn_imputer_neighbors"]
        self.knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('knn_imputer', self.knn_imputer, self.num_features),
                ('simple_imputer', self.simple_imputer, self.cat_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    @staticmethod
    def fill_in_constant(column, value):
        return column.fillna(value)

    def fit(self, x):
        self.column_transformer.fit(x)
        return self

    def transform(self, x):
        x_copy = x.copy()
        if 'children' in self.num_features:
            x_copy['children'] = self.fill_in_constant(x_copy['children'],
                                        self.config["children_impute_value"])
        x_transformed = self.column_transformer.transform(x_copy)
        return x_transformed
