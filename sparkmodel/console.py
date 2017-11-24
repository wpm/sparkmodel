"""
Command line interface for Sparkmodel.
"""

import logging

import click as click
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.utils import AnalysisException

from sparkmodel import __version__
from .main import train_and_save_pipeline, generate_and_save_data_set, spark


class SparkPathType(click.ParamType):
    name = "spark-path"

    def __init__(self, should_exist=True):
        self.should_exist = should_exist

    def convert(self, value, param, ctx):
        exists = self.spark_path_exists(value)
        if self.should_exist and not exists:
            self.fail(f"output path {value} does not exist")
        elif not self.should_exist and exists:
            self.fail(f"output path {value} already exists")
        else:
            return value

    @staticmethod
    def spark_path_exists(path):
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


@click.group()
@click.version_option(version=__version__)
@click.option("--log", type=click.Choice(["debug", "info", "warning", "error", "critical"]), default="warning",
              help="Logging level")
def main(log: str):
    """Machine learning command line framework"""
    log_level = log.upper()
    logging.basicConfig(format="%(asctime)s:%(levelname)s Sparkmodel:%(message)s", level=getattr(logging, log_level))


@click.group(short_help="Train a model", invoke_without_command=True)
def train():
    """Train a predictor on DATA and save it as MODEL.
    """
    pass


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
    return train_and_save_pipeline(model_path, data_path, LinearSVC, params)


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
    return train_and_save_pipeline(model_path, data_path, LogisticRegression, params)


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


train.add_command(train_svm_command, name="svm")
train.add_command(train_logistic_regression_command, name="logistic-regression")

main.add_command(train)
main.add_command(predict_command, name="predict")
main.add_command(generate_command, name="generate")
