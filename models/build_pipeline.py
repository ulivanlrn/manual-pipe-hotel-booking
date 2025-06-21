# import logging
from sklearn.pipeline import Pipeline
from preprocessing.feature_encoding import Encoder

def build_pipeline(model_config):
    steps = []

    # encoding
    if model_config["encoding"]["requires_encoding"]:
        random_state = model_config["encoding"]["random_state"]
        card_threshold = model_config["encoding"]["cardinality_threshold"]
        encoder = Encoder(random_state=random_state, threshold=card_threshold)
        steps.append(('encoder', encoder))

    return Pipeline(steps)