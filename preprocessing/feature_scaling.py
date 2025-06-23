from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer

class Scaler(BaseEstimator, TransformerMixin):
    """
        A custom scaler, which applies different techniques to different subsets of features.
        Controlled from the configuration file.

        Attributes:
        scaling_config (dict): Scaling configuration.
    """
    def __init__(self, scaling_config):
        self.scaling_config = scaling_config
        self.transformers = []

        # standard scaling
        if self.scaling_config["std_scaling"]["flag"]:
            self.std_scaling = ('std_scaling',
                                StandardScaler(),
                                self.scaling_config["std_scaling"]["columns"])
            self.transformers.append(self.std_scaling)

        # min max scaling
        if self.scaling_config["min_max_scaling"]["flag"]:
            self.min_max_scaling = ('min_max_scaling',
                                    MinMaxScaler(),
                                    self.scaling_config["min_max_scaling"]["columns"])
            self.transformers.append(self.min_max_scaling)

        # constructing ColumnTransformer
        self.column_transformer = ColumnTransformer(
            transformers=self.transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    def fit(self, x, _):
        self.column_transformer.fit(x)
        return self

    def transform(self, x):
        x_transformed = self.column_transformer.transform(x)
        return x_transformed