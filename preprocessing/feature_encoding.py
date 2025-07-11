from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

def separate_by_cardinality(df, threshold):
    """
    Function which figures out the names of categorical features and divides them by cardinality.

    :param df: DataFrame.
    :param threshold: Threshold that separates high a low cardinality features.
    :return: Tuple consisting of two lists, with names of high cardinality features
    and names of low cardinality features respectively.
    """
    info_cat_feats = df.select_dtypes('object').nunique()
    cat_feats = info_cat_feats.index
    high_cardinality_feats = info_cat_feats[info_cat_feats > threshold].index
    low_cardinality_feats = cat_feats.difference(high_cardinality_feats)
    return list(high_cardinality_feats), list(low_cardinality_feats)

class Encoder(BaseEstimator, TransformerMixin):
    """
    A custom categorical encoder, which applies OneHotEncoder to low cardinality features and
    TargetEncoder to high cardinality features.

    Attributes:
        encoding_config (dict): Encoding configuration.
    """
    def __init__(self, encoding_config):
        self.encoding_config = encoding_config
        self.random_state = self.encoding_config["random_state"]
        self.threshold = self.encoding_config["cardinality_threshold"]
        self.one_hot_encoder = OneHotEncoder(drop='first',
                                             sparse_output=False,
                                             handle_unknown='ignore')
        self.target_encoder = TargetEncoder(target_type='binary',
                                            random_state=self.random_state)
        self.column_transformer = None
        self.low_card_feats = None
        self.high_card_feats = None

    def fit(self, x, y):
        self.high_card_feats, self.low_card_feats = separate_by_cardinality(x, self.threshold)
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('one_hot', self.one_hot_encoder, self.low_card_feats),
                ('te', self.target_encoder, self.high_card_feats)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        self.column_transformer.fit(x, y)
        return self

    def transform(self, x):
        x_transformed = self.column_transformer.transform(x)
        return x_transformed



