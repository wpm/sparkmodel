import logging

import pandas
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def train_model(features, labels):
    """
    Train a model

    :param features: sample features
    :type features: [[int]]
    :param labels: labels for the features
    :type labels: [int]
    :return: trained model
    :rtype: SVC
    """
    logging.info(f"Train with {len(features)} instances.")
    assert len(features) == len(labels)
    return SVC().fit(features, labels)


def predict_labels(model, features):
    """
    Predict labels for data using a trained model

    :param model: model to use
    :type model: SVC
    :param features: sample features
    :type features: [[int]]
    :return: predicted labels
    :rtype: [int]
    """
    logging.info(f"Predict {len(features)} labels.")
    return model.predict(features)


def generate_data(n):
    """
    Generate random training data

    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1).

    :param n: number of data points to generate
    :type n: int
    :return: set of (x, y) coordinates and labels
    :rtype: DataFrame
    """
    logging.info(f"Generate {n} data points.")
    x, y = make_blobs(n, centers=[(-1, -1), (1, 1)])
    return pandas.DataFrame(data={"x": x[:, 0], "y": x[:, 1], "label": y})
