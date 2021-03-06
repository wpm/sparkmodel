import logging
import re

from pyspark.ml import Pipeline

from sparkmodel import spark


def train_and_save_pipeline(model_path: str, data_path: str, estimator_class: type, params: dict = ()):
    """
    Train and save a pipeline containing a single estimator with the specified parameters.

    This is the action called from the command line handler for a train subcommand.

    :param model_path: where to save the trained pipeline model
    :param data_path: data on which to train the model
    :param estimator_class: the estimator to train
    :param params: dictionary of optional estimator parameters
    """

    def snake_keys_to_camel(snake_dict: dict):
        def snake_to_camel(text: str):
            return re.sub(r"_([a-zA-Z0-9])", lambda m: m.group(1).upper(), text)

        return dict((snake_to_camel(k), v) for k, v in snake_dict.items())

    data = spark().read.load(data_path)
    # Command names taken from Click parameters are lowercase underscore-delimited "snake-case" strings, while Spark
    # estimators take camel-case parameters.
    params = snake_keys_to_camel(dict(params))
    estimator = estimator_class().setParams(**params)
    pipeline = Pipeline(stages=[estimator]).fit(data)
    pipeline.save(model_path)
    logging.info(f"""Created pipeline model in {model_path}""")
