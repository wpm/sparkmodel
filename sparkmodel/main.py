import logging

import click as click
import pandas as pandas
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from sparkmodel import __version__
from .model import train_model, predict_labels, generate_data


@click.group()
@click.version_option(version=__version__)
@click.option("--log", type=click.Choice(["debug", "info", "warning", "error", "critical"]), default="warning",
              help="Logging level")
def main(log):
    """Machine learning command line framework"""
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=getattr(logging, log.upper()))


@click.command(short_help="Train a model")
@click.argument("model_file", type=click.File("wb"), metavar="MODEL")
@click.argument("data_file", type=click.File(), metavar="DATA")
def train(model_file, data_file):
    """Train an SVM model on DATA and save it as MODEL.

    The data is a csv file that contains a 'label' column. All other columns are treated as features.
    """
    data = pandas.read_csv(data_file)
    model = train_model(features(data), data.label)
    joblib.dump(model, model_file)
    logging.info(f"Created model in {model_file.name}")


@click.command(short_help="Predict labels")
@click.argument("model_file", type=click.File("rb"), metavar="MODEL")
@click.argument("data_file", type=click.File(), metavar="DATA")
@click.option("--labeled-data", type=click.File("w"))
def predict(model_file, data_file, labeled_data):
    """Use MODEL to make label predictions for DATA.

    Optionally output data with a 'predict' column containing label predictions.
    If a 'label' column in present in the data, calculate and print the accuracy.
    """
    model = joblib.load(model_file)
    data = pandas.read_csv(data_file)
    labels = predict_labels(model, features(data))
    if labeled_data:
        data["predict"] = labels
        data.to_csv(labels, index=False)
    if "label" in data:
        click.echo(f"Accuracy {accuracy_score(data.label, labels):0.4f}")


@click.command(short_help="Generate data")
@click.option("--n", default=1000, help="number of data points to generate")
@click.option("--output-file", type=click.File("w"), default="-")
def generate(n, output_file):
    """
    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1).
    """
    click.echo(generate_data(n).to_csv(output_file, index=False))


main.add_command(train)
main.add_command(predict)
main.add_command(generate)


def features(data):
    return data.drop("label", axis="columns")
