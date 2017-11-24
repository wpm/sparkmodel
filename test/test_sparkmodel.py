import os
import tempfile
import unittest

from click.testing import CliRunner
from pyspark.ml.classification import LogisticRegression

from main import train_and_save_pipeline, generate_and_save_data_set, spark
from sparkmodel.console import main


class TestUtilityMixin(object):
    temporary_directory = None

    @classmethod
    def create_temporary_directory(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(cls.temporary_directory.name)

    @classmethod
    def temporary_filename(cls, name):
        return os.path.join(cls.temporary_directory.name, name)

    @classmethod
    def generate_data_file(cls, basename, n):
        filename = cls.temporary_filename(basename)
        generate_and_save_data_set(n, filename)
        return filename


class TestModels(unittest.TestCase, TestUtilityMixin):
    train_data = None

    @classmethod
    def setUpClass(cls):
        cls.create_temporary_directory()
        cls.train_data = cls.generate_data_file("train-data", 100)

    @classmethod
    def tearDownClass(cls):
        # Ensure that Spark does its temporary file cleanup before the temporary directory is deleted.
        spark().stop()

    def test_train_pipeline(self):
        model = self.temporary_filename("model")
        train_and_save_pipeline(model, self.train_data, LogisticRegression)
        self.assertTrue(os.path.isdir(model))


class TestCommandLine(unittest.TestCase, TestUtilityMixin):
    train_data = test_data = None

    @classmethod
    def setUpClass(cls):
        cls.create_temporary_directory()
        cls.runner = CliRunner()
        cls.train_data = cls.generate_data_file("train-data", 1000)
        cls.test_data = cls.generate_data_file("test-data", 100)

    @classmethod
    def tearDownClass(cls):
        # Ensure that Spark does its temporary file cleanup before the temporary directory is deleted.
        spark().stop()

    def test_version(self):
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)

    def test_svm(self):
        self._test_train_command("svm")

    def test_logistic_regression(self):
        self._test_train_command("logistic-regression")

    def _test_train_command(self, command):
        model = self.temporary_filename(f"model.{command}")
        output = self.temporary_filename(f"output.{command}")
        result = self.runner.invoke(main, ["train", command, model, self.train_data])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(model))
        result = self.runner.invoke(main, ["predict", model, self.test_data, f"""--labeled-data={output}"""])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(output))
        self.assertRegexpMatches(result.output, "Accuracy 0.\d\d\d\d")

    def test_generate(self):
        generated_data = self.temporary_filename("generated-data")
        result = self.runner.invoke(main, ["generate", generated_data])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(generated_data))

    def test_output_path_exists(self):
        output = self.temporary_filename("output")
        result = self.runner.invoke(main, ["generate", output])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(output))
        result = self.runner.invoke(main, ["generate", output])
        self.assertEqual(2, result.exit_code, msg=result.output)
        self.assertRegexpMatches(result.output, ".+/output already exists")

    def test_input_path_does_not_exist(self):
        result = self.runner.invoke(main, ["predict", self.temporary_filename("model"), self.test_data])
        self.assertEqual(2, result.exit_code, msg=result.output)
        self.assertRegexpMatches(result.output, ".+/model does not exist")
