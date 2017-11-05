import logging

import pandas
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

__version__ = "1.0.0"


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


def generate_data(n):
    """
    Generate random training data

    :param n: number of data points to generate
    :type n: int
    :return: set of (x, y) coordinates and labels
    :rtype: DataFrame
    """
    print("SSS")
    x, y = make_blobs(n, centers=[(-1, -1), (1, 1)])
    return pandas.DataFrame(data={"x": x[:, 0], "y": x[:, 1], "label": y})
