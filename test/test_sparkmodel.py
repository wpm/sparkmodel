import tempfile
import unittest

import os
from click.testing import CliRunner

from sparkmodel.main import main, spark, generate


class TestCommandLine(unittest.TestCase):
    temporary_directory = train_data = test_data = None

    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(cls.temporary_directory.name)
        cls.runner = CliRunner()
        cls.train_data = cls.temporary_filename("train-data")
        cls.test_data = cls.temporary_filename("test-data")
        generate(1000, cls.train_data)
        generate(100, cls.test_data)

    @classmethod
    def tearDownClass(cls):
        spark().stop()

    @classmethod
    def temporary_filename(cls, name):
        return os.path.join(cls.temporary_directory.name, name)

    def test_version(self):
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)

    def test_svm(self):
        self.train_command("svm")

    def test_logistic_regression(self):
        self.train_command("logistic-regression")

    def test_generate(self):
        data = self.temporary_filename("generated-data")
        result = self.runner.invoke(main, ["generate", data])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(data))

    def train_command(self, command):
        model = self.temporary_filename(f"model.{command}")
        output = self.temporary_filename(f"output.{command}")
        result = self.runner.invoke(main, ["train", command, model, self.train_data])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(model))
        result = self.runner.invoke(main, ["predict", model, self.test_data, f"""--labeled-data={output}"""])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.isdir(output))
        self.assertRegexpMatches(result.output, "Accuracy 0.\d\d\d\d")
