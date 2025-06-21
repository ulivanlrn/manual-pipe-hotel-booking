# import logging
from sklearn.pipeline import Pipeline
from preprocessing.feature_encoding import Encoder
from preprocessing.handling_outliers import OutlierHandler

def build_pipeline(model_config):
    steps = []

    # encoding
    if model_config["encoding"]["requires_encoding"]:
        random_state = model_config["encoding"]["random_state"]
        card_threshold = model_config["encoding"]["cardinality_threshold"]
        encoder = Encoder(random_state=random_state, threshold=card_threshold)
        steps.append(('encoder', encoder))

    # outliers
    if model_config["outliers"]["flag"]:

        outlier_handler = OutlierHandler()
        steps.append(('outlier_handler', outlier_handler))

    return Pipeline(steps)