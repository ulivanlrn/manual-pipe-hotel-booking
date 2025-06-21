import logging
from sklearn.pipeline import Pipeline
from preprocessing.feature_encoding import Encoder

def build_pipeline(model_config):
    steps = []
    if model_config["needs_encoding"]:
        steps.append(('encoder', Encoder()))
    logging.info("check")
    return Pipeline(steps)