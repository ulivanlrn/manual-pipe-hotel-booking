from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(df):
    for col in df:
        df[col] = np.log1p(df[col])
    return df.fillna(0)

class OutlierHandler(BaseEstimator, TransformerMixin):

    def __init__(self, outlier_config):
        self.outlier_config = outlier_config
        self.transformers = []
        if self.outlier_config["log_transform"]["flag"]:
            self.log_transform = ('log_transform', FunctionTransformer(log_transform),
                                  self.outlier_config["log_transform"]["columns"])
            self.transformers.append(self.log_transform)
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