# import logging
from sklearn.pipeline import Pipeline
from preprocessing.feature_encoding import Encoder
from preprocessing.feature_scaling import Scaler
from preprocessing.handling_outliers import OutlierHandler

def build_pipeline(model_config):
    steps = []

    # encoding
    if model_config["encoding"]["requires_encoding"]:
        encoding_config = model_config["encoding"]
        encoder = Encoder(encoding_config=encoding_config)
        steps.append(('encoder', encoder))

    # outliers
    if model_config["outliers"]["flag"]:
        outlier_config = model_config["outliers"]
        outlier_handler = OutlierHandler(outlier_config=outlier_config)
        steps.append(('outlier_handler', outlier_handler))

    # scaling
    if model_config["scaling"]["flag"]:
        scaling_config = model_config["scaling"]
        scaler = Scaler(scaling_config=scaling_config)
        steps.append(('scaler', scaler))

    return Pipeline(steps)