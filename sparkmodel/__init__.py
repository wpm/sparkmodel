import logging

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import make_blobs

__version__ = "1.0.0"


def spark() -> SparkSession:
    return SparkSession.builder.appName("Sparkmodel").getOrCreate()


def generate_and_save_data_set(n: int, output_path: str):
    x, y = make_blobs(n, centers=[(-1, -1), (1, 1)])
    samples = [(int(label), Vectors.dense(features)) for label, features in zip(y, x)]
    data = spark().createDataFrame(samples, schema=["label", "features"])
    data.write.save(output_path)
    logging.info(f"Generated data set with {n} samples in {output_path}.")
