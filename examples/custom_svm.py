import click as click
from pyspark.ml.classification import LinearSVC

from console import SparkOutputPath, SparkInputPath
from main import train_and_save_pipeline


@click.command(short_help="custom support vector machine")
@click.argument("model_path", type=SparkOutputPath(), metavar="MODEL")
@click.argument("data_path", type=SparkInputPath(), metavar="DATA")
@click.option("--max-iter", default=50, help="max number of iterations (>= 0).")
def main(model_path: str, data_path: str, max_iter: int):
    """Train a support vector machine on DATA and save it as MODEL.
    """
    return train_and_save_pipeline(model_path, data_path, LinearSVC, {"maxIter": max_iter})


def train_commands():
    return [(main, "custom-svm")]


if __name__ == "__main__":
    main()
