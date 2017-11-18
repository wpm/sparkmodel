import logging

import click as click
from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import make_blobs

from sparkmodel import __version__


def spark():
    return SparkSession.builder.appName("Sparkmodel").getOrCreate()


@click.group()
@click.version_option(version=__version__)
@click.option("--log", type=click.Choice(["debug", "info", "warning", "error", "critical"]), default="warning",
              help="Logging level")
def main(log):
    """Machine learning command line framework"""
    log_level = log.upper()
    logging.basicConfig(format="%(asctime)s:%(levelname)s Sparkmodel:%(message)s", level=getattr(logging, log_level))


@click.command(short_help="Train a model")
@click.argument("model_path", metavar="MODEL")
@click.argument("data_path", metavar="DATA")
def train(model_path, data_path):
    """Train an SVM model on DATA and save it as MODEL.
    """
    data = spark().read.load(data_path)
    model = LinearSVC().fit(data)
    model.save(model_path)
    logging.info(f"Created model in {model_path}")


@click.command(short_help="Predict labels")
@click.argument("model_path", metavar="MODEL")
@click.argument("data_path", metavar="DATA")
@click.option("--labeled-data", metavar="OUTPUT", help="file to which to write labeled data")
def predict(model_path, data_path, labeled_data):
    """Use MODEL to make label predictions for DATA.

    Optionally output data with a 'predict' column containing label predictions.
    If a 'label' column in present in the data, calculate and print the accuracy.
    """
    data = spark().read.load(data_path)
    model = LinearSVCModel.load(model_path)
    data = model.transform(data)
    if labeled_data:
        data.drop("features").write.save(labeled_data)
    if "label" in data.columns:
        accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(data)
        click.echo(f"Accuracy {accuracy:0.4f}")


@click.command(short_help="Generate data")
@click.argument("output_path", metavar="OUTPUT")
@click.option("--n", default=1000, help="number of data points to generate")
def generate(n, output_path):
    """
    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1) and write it to OUTPUT.
    """
    x, y = generate_data(n)
    samples = [(int(label), Vectors.dense(features)) for label, features in zip(y, x)]
    data = spark().createDataFrame(samples, schema=["label", "features"])
    data.write.save(output_path)


main.add_command(train)
main.add_command(predict)
main.add_command(generate)


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
    return x, y
