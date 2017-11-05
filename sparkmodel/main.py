import logging

import click as click
import pandas as pandas
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from sparkmodel import predict_labels, __version__, generate_data
from . import train_model


@click.group()
@click.version_option(version=__version__)
@click.option("--log", type=click.Choice(["debug", "info", "warning", "error", "critical"]), default="warning",
              help="logging level")
def main(log):
    """Machine learning command line framework"""
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=getattr(logging, log.upper()))


@click.command()
@click.argument("model_file", type=click.File("wb"))
@click.argument("data_file", type=click.File())
def train(model_file, data_file):
    """Train a model

    Train an SVM model.
    """
    data = pandas.read_csv(data_file)
    model = train_model(data)
    joblib.dump(model, model_file)
    logging.info(f"Created model in {model_file.name}")


@click.command()
@click.argument("model_file", type=click.File("rb"))
@click.argument("data_file", type=click.File())
@click.option("--labeled-data", type=click.File("w"))
def predict(model_file, data_file, labeled_data):
    """Use a model to make predictions

    Optionally output data with a 'predict' column containing label predictions.
    If a 'label' column in present in the data, calculate and print the accuracy.
    """
    model = joblib.load(model_file)
    data = pandas.read_csv(data_file)
    labels = predict_labels(model, data)
    if labeled_data:
        data["predict"] = labels
        data.to_csv(labeled_data, index=False)
    if "label" in data:
        click.echo(f"Accuracy {accuracy_score(data.label, labels):0.4f}")


@click.command()
@click.option("--n", default=1000, help="number of data points to generate")
@click.option("--data-file", type=click.File("w"), default="-")
def generate(n, data_file):
    """
    Generate sample data

    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1).
    """
    click.echo(generate_data(n).to_csv(data_file, index=False))


main.add_command(train)
main.add_command(predict)
main.add_command(generate)
