import logging
from sklearn.svm import SVC


def train_model(data):
    """
    Train a model

    :param data: labeled training data
    :type data: DataFrame
    :return: a model trained on the data
    :rtype: SVC
    """
    logging.info(f"Train with {len(data)} instances.")
    return SVC().fit(data.drop("label", axis="columns"), data.label)


def predict_labels(model, data):
    """
    Predict labels for data using a trained model

    :param model: model to use
    :type model: SVC
    :param data: data to predict labels for
    :type data: DataFrame
    :return: data with associated labels
    :rtype: array
    """
    logging.info(f"Predict {len(data)} labels.")
    return model.predict(data.drop("label", axis="columns"))
