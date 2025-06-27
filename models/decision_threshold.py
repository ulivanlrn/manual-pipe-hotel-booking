from sklearn.model_selection import FixedThresholdClassifier
import logging

def tune_threshold(model, threshold):
    """
    Wrap the model into FixedThresholdClassifier.

    :param model: Model.
    :param threshold:
    :return: Fixed threshold classifier.
    """
    model = FixedThresholdClassifier(model, threshold=threshold)
    logging.info("The decision threshold is set to {}".format(threshold))
    return model
