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
        self.runner.invoke(main, ["generate", "train-data"])
        self.runner.invoke(main, ["generate", "--n=100", "test-data"])
        result = self.runner.invoke(main, ["train", "model", "train-data"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        result = self.runner.invoke(main, ["predict", "model", "test-data", "--labeled-data=test-output"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertRegexpMatches(result.output, "Accuracy 0.\d\d\d\d")
