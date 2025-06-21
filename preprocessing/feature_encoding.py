from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

def separate_on_cardinality(df):
    info_cat_feats = df.select_dtypes('object').nunique()
    cat_feats = info_cat_feats.index
    high_cardinality_feats = info_cat_feats[info_cat_feats > 5].index
    low_cardinality_feats = cat_feats.difference(high_cardinality_feats)
    return list(high_cardinality_feats), list(low_cardinality_feats)

class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, random_state, threshold):
        self.random_state = random_state
        self.threshold = threshold
        self.one_hot_encoder = OneHotEncoder(drop='first',
                                             sparse_output=False,
                                             handle_unknown='ignore')
        self.target_encoder = TargetEncoder(target_type='binary',
                                            random_state=self.random_state)
        self.column_transformer = None
        self.low_card_feats = None
        self.high_card_feats = None

    def fit(self, x, y):
        self.high_card_feats, self.low_card_feats = separate_on_cardinality(x)
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



