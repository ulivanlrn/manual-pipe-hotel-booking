from sklearn.model_selection import FixedThresholdClassifier
import logging

def tune_threshold(model, model_config):
    """
    Wrap the model into FixedThresholdClassifier.

    :param model: Model.
    :param model_config: Model configuration.
    :return: Fixed threshold classifier.
    """
    threshold = model_config["decision_threshold_tuning"]["threshold"]
    model = FixedThresholdClassifier(model, threshold=threshold)
    logging.info("The decision threshold is set to {}".format(threshold))
    return model
