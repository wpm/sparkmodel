import logging

import click as click
from numpy import array
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LinearSVC, LogisticRegression
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
def main(log: str):
    """Machine learning command line framework"""
    log_level = log.upper()
    logging.basicConfig(format="%(asctime)s:%(levelname)s Sparkmodel:%(message)s", level=getattr(logging, log_level))


@click.group(short_help="Train a model", invoke_without_command=True)
def train():
    """Train a predictor on DATA and save it as MODEL.
    """
    pass


def train_pipeline(model_path: str, data_path: str, estimator):
    data = spark().read.load(data_path)
    pipeline = Pipeline(stages=[estimator]).fit(data)
    pipeline.save(model_path)
    logging.info(f"""Created pipeline model in {model_path}""")


@click.command(short_help="support vector machine")
@click.argument("model_path", metavar="MODEL")
@click.argument("data_path", metavar="DATA")
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
def train_svm_command(model_path: str, data_path: str,
                      aggregation_depth: int, fit_intercept: bool, max_iter: int, reg_param: float,
                      standardization: bool,
                      threshold: float, tol: float):
    """Train a support vector machine on DATA and save it as MODEL.
    """
    svm_estimator = LinearSVC(aggregationDepth=aggregation_depth, fitIntercept=fit_intercept, maxIter=max_iter,
                              regParam=reg_param, standardization=standardization, threshold=threshold, tol=tol)
    return train_pipeline(model_path, data_path, svm_estimator)


train.add_command(train_svm_command, name="svm")


@click.command(short_help="logistic regression")
@click.argument("model_path", metavar="MODEL")
@click.argument("data_path", metavar="DATA")
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
def train_logistic_regression_command(model_path: str, data_path: str,
                                      max_iter: int, reg_param: float, elastic_net_param: float, tol: float,
                                      threshold: float,
                                      standardization: bool, aggregation_depth: int, family: str):
    """Train a logistic regression predictor on DATA and save it as MODEL.
    """
    logistic_regression_estimator = LogisticRegression(maxIter=max_iter, regParam=reg_param,
                                                       elasticNetParam=elastic_net_param,
                                                       tol=tol,
                                                       threshold=threshold, standardization=standardization,
                                                       aggregationDepth=aggregation_depth, family=family)
    return train_pipeline(model_path, data_path, logistic_regression_estimator)


train.add_command(train_logistic_regression_command, name="logistic-regression")


@click.command(short_help="Predict labels")
@click.argument("model_path", metavar="MODEL")
@click.argument("data_path", metavar="DATA")
@click.option("--labeled-data", metavar="OUTPUT", help="file to which to write labeled data")
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
@click.argument("output_path", metavar="OUTPUT")
@click.option("--n", default=1000, help="number of data points to generate")
def generate_command(n: int, output_path: str):
    """
    Generate (x,y) data in Gaussian distributions around the points (-1, -1) and (1,1) and write it to OUTPUT.
    """
    generate(n, output_path)


def generate(n: int, output_path: str):
    logging.info(f"Generate {n} data points.")
    x, y = make_blobs(n, centers=[(-1, -1), (1, 1)])
    samples = [(int(label), Vectors.dense(features)) for label, features in zip(y, x)]
    data = spark().createDataFrame(samples, schema=["label", "features"])
    data.write.save(output_path)


main.add_command(train)
main.add_command(predict_command, name="predict")
main.add_command(generate_command, name="generate")
