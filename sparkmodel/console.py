"""
Command line interface for Sparkmodel.
"""

import logging
import os
from importlib import import_module
from typing import List, Tuple, Callable

import click as click
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.utils import AnalysisException

from sparkmodel import __version__, generate_and_save_data_set, spark
from .train import train_and_save_pipeline


class SparkPathType(click.ParamType):
    name = "spark-path"

    def __init__(self, should_exist=True):
        self.should_exist = should_exist

    def convert(self, path: str, _, __) -> str:
        exists = self.spark_path_exists(path)
        if self.should_exist and not exists:
            self.fail(f"output path {path} does not exist")
        elif not self.should_exist and exists:
            self.fail(f"output path {path} already exists")
        else:
            return path

    @staticmethod
    def spark_path_exists(path: str) -> bool:
        try:
            spark().read.load(path)
        except AnalysisException as e:
            return not e.desc.startswith("Path does not exist")
        return True


class SparkOutputPath(SparkPathType):
    def __init__(self):
        super().__init__(should_exist=False)


class SparkInputPath(SparkPathType):
    def __init__(self):
        super().__init__(should_exist=True)


# noinspection PyClassHasNoInit
class CustomModel(click.ParamType):
    name = "custom-model"

    def convert(self, file_path: str, _, __) -> List[Tuple[str, Callable]]:
        if not os.path.isfile(file_path):
            self.fail(f"{file_path} is not a file.")
        module_path = ".".join(os.path.split(file_path[:-3]))
        try:
            return import_module(module_path).train_commands
        except ModuleNotFoundError:
            self.fail("Could not load custom model definitions from {file_path}.")
        except AttributeError:
            self.fail(f"{file_path} is missing a train_commands definition.")


@click.group()
@click.version_option(version=__version__)
@click.option("--log", type=click.Choice(["debug", "info", "warning", "error", "critical"]), default="warning",
              help="Logging level")
@click.option("--custom-model", type=CustomModel(), help="Python file containing custom models")
@click.pass_context
def main(ctx, log: str, custom_model: List[Tuple[str, Callable]]):
    """Machine learning command line framework"""
    log_level = log.upper()
    logging.basicConfig(format="%(asctime)s:%(levelname)s Sparkmodel:%(message)s", level=getattr(logging, log_level))
    if custom_model is not None:
        ctx.obj = custom_model
    else:
        ctx.obj = []


@click.command(short_help="support vector machine")
@click.argument("model_path", type=SparkOutputPath(), metavar="MODEL")
@click.argument("data_path", type=SparkInputPath(), metavar="DATA")
@click.option("--aggregation-depth", default=2, help="suggested depth for treeAggregate (>= 2).")
@click.option("--fit-intercept", default=True, help="whether to fit an intercept term.")
@click.option("--max-iter", default=100, help="max number of iterations (>= 0).")
@click.option("--reg-param", default=0.0, help="regularization parameter (>= 0).")
@click.option("--standardization", default=True,
              help="whether to standardize the training features before fitting the model.")
@click.option("--threshold", default=0.0,
              help="The threshold in binary classification applied to the linear model prediction.  This threshold "
                   "can be any real number, where Inf will make all predictions 0.0 and -Inf will make all "
                   "predictions 1.0.")
@click.option("--tol", default=1e-6, help="the convergence tolerance for iterative algorithms (>= 0).")
def train_svm_command(model_path: str, data_path: str, **params: dict):
    """Train a support vector machine on DATA and save it as MODEL.
    """
    train_and_save_pipeline(model_path, data_path, LinearSVC, params)


@click.command(short_help="logistic regression")
@click.argument("model_path", type=SparkOutputPath(), metavar="MODEL")
@click.argument("data_path", type=SparkInputPath(), metavar="DATA")
@click.option("--max-iter", default=100, help="max number of iterations (>= 0).")
@click.option("--reg-param", default=0.0, help="regularization parameter (>= 0).")
@click.option("--elastic-net-param", default=0.0,
              help="the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. "
                   "For alpha = 1, it is an L1 penalty.")
@click.option("--tol", default=1e-6, help="the convergence tolerance for iterative algorithms (>= 0).")
@click.option("--threshold", default=0.5,
              help="Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are "
                   "both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p].")
@click.option("--standardization", default=True,
              help="whether to standardize the training features before fitting the model.")
@click.option("--aggregation-depth", default=2, help="suggested depth for treeAggregate (>= 2).")
@click.option("--family", type=click.Choice(["auto", "binomial", "multinomial"]), default="auto",
              help="The name of family which is a description of the label distribution to be used in the model. "
                   "Supported options: auto, binomial, multinomial")
def train_logistic_regression_command(model_path: str, data_path: str, **params: dict):
    """Train a logistic regression predictor on DATA and save it as MODEL.
    """
    train_and_save_pipeline(model_path, data_path, LogisticRegression, params)


class Train(click.MultiCommand):
    COMMANDS = [("svm", train_svm_command), ("logistic-regression", train_logistic_regression_command)]

    def __init__(self, **attrs):
        super().__init__("train", **attrs)

    def list_commands(self, ctx):
        return [name for (name, _) in ctx.obj + self.COMMANDS]

    def get_command(self, ctx, name):
        return next(c for (n, c) in ctx.obj + self.COMMANDS if n == name)


@click.command(short_help="Predict labels")
@click.argument("model_path", type=SparkInputPath(), metavar="MODEL")
@click.argument("data_path", type=SparkInputPath(), metavar="DATA")
@click.option("--labeled-data", type=SparkOutputPath(), metavar="OUTPUT", help="destination path for labeled data")
def predict_command(model_path: str, data_path: str, labeled_data: str):
    """Use MODEL to make label predictions for DATA.

    Optionally output data with a 'predict' column containing label predictions.
    If a 'label' column in present in the data, calculate and print the accuracy.
    """
    data = spark().read.load(data_path)
    pipeline = PipelineModel.load(model_path)
    data = pipeline.transform(data)
    if labeled_data:
        data.drop("features").write.save(labeled_data)
    if "label" in data.columns:
        accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(data)
        click.echo(f"Accuracy {accuracy:0.4f}")


@click.command(short_help="Generate data")
@click.argument("output_path", type=SparkOutputPath(), metavar="OUTPUT")
@click.option("--n", default=1000, help="number of data points to generate")
def generate_command(n: int, output_path: str):
    """
    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1) and write it to OUTPUT.
    """
    generate_and_save_data_set(n, output_path)


main.add_command(Train(help="Train a model"))
main.add_command(predict_command, name="predict")
main.add_command(generate_command, name="generate")
