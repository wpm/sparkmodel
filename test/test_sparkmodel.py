import unittest

from click.testing import CliRunner

from sparkmodel.main import main


class TestCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)

    def test_model(self):
        with self.runner.isolated_filesystem():
            self.runner.invoke(main, ["generate", "train-data"])
            self.runner.invoke(main, ["generate", "--n=100", "test-data"])
            result = self.runner.invoke(main, ["train", "svm", "model", "train-data"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            result = self.runner.invoke(main, ["predict", "model", "test-data", "--labeled-data=test-output"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertRegexpMatches(result.output, "Accuracy 0.\d\d\d\d")
