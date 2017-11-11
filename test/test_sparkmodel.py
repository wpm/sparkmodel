import os
import unittest

from click.testing import CliRunner

from sparkmodel.main import main


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)

    def test_model(self):
        with self.runner.isolated_filesystem():
            self.runner.invoke(main, ["generate", "--output-file=train.csv"])
            self.runner.invoke(main, ["generate", "--n=100", "--output-file=test.csv"])
            result = self.runner.invoke(main, ["train", "model.pk", "train.csv"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            result = self.runner.invoke(main, ["predict", "model.pk", "test.csv", "--labeled-data=output.csv"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertRegexpMatches(result.output, "Accuracy 0.\d\d\d\d")
            self.assertTrue(os.path.exists("output.csv"))
