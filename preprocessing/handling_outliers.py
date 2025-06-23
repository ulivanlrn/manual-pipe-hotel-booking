from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def log_transform(df):
    """
    Apply logarithmic transformation to all the columns in a dataframe.

    :param df: Original dataframe.
    :return: Transformed dataframe.
    """
    for col in df:
        df[col] = np.log1p(df[col])
    return df.fillna(0)

def outliers_substitute(df, config):
    """
    Apply manual substitution of values in every column in a dataframe with np.where function.
    The thresholds corresponding to each column are given in values variable.

    :param df: Original dataframe.
    :param config: Configuration file.
    :return: DataFrame with outliers replaced.
    """
    columns = config["outliers_substitute"]["columns"]
    values = config["outliers_substitute"]["values"]
    for i in range(len(columns)):
        df[columns[i]] = np.where(df[columns[i]]>=values[i], values[i], df[columns[i]])
    return df

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
        A custom outlier handler, which applies different techniques to different subsets of features.
        Controlled from the configuration file.

        Attributes:
            outlier_config (dict): Outliers configuration.
    """
    def __init__(self, outlier_config):
        self.outlier_config = outlier_config
        self.transformers = []

        # log_transform
        if self.outlier_config["log_transform"]["flag"]:
            self.log_transform = ('log_transform',
                                  FunctionTransformer(log_transform),
                                  self.outlier_config["log_transform"]["columns"])
            self.transformers.append(self.log_transform)

        # manual substitution
        if self.outlier_config["outliers_substitute"]["flag"]:
            self.outliers_substitute = ('outliers_substitute',
                                        FunctionTransformer(lambda df: outliers_substitute(df, self.outlier_config)),
                                        self.outlier_config["outliers_substitute"]["columns"])
            self.transformers.append(self.outliers_substitute)

        # discretization
        if self.outlier_config["discretization"]["flag"]:
            self.discretizer = KBinsDiscretizer(n_bins=self.outlier_config["discretization"]["n_bins"],
                                                random_state=self.outlier_config["discretization"]["random_state"],
                                                encode="ordinal", strategy="quantile"
                                                )
            self.discretization = ('discretization',
                                   self.discretizer,
                                   self.outlier_config["discretization"]["columns"])
            self.transformers.append(self.discretization)

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