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
            self.log_transform = ('log_transform', FunctionTransformer(),
                                  self.outlier_config["log_transform"]["columns"])
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        pass