from sklearn.pipeline import Pipeline
from preprocessing.feature_encoding import Encoder
from preprocessing.feature_scaling import Scaler
from preprocessing.handling_outliers import OutlierHandler
from sklearn.linear_model import LogisticRegression

def get_model_class(name):
    """
    Function to get a model class by name.
    :param name: Name provided in config under type.
    :return: Model class.
    """
    if name == "LogisticRegression":
        return LogisticRegression
    else:
        raise ValueError(f"Unsupported model type: {name}")

def build_pipeline(model_config):
    """
    Function to build a pipeline.
    :param model_config: Configuration file which controls what should be included in the pipeline.
    :return: A pipeline object.
    """
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

    # model
    model_class = get_model_class(model_config["type"])
    model = model_class(**model_config["params"])
    steps.append(('model', model))

    return Pipeline(steps)