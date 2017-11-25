# Spark Model

A minimal command line tool to train models using Spark.

See

    sparkmodel --help

for usage instructions.

Setup also installs a `sparkmodel_submit.py` on your path which may be used as the Python script for a
`spark-submit` command.


### Custom Commands

The Spark Model command framework also supports custom models.
Custom models are implemented in Python source files that export a `train_commands` variable, which
is a list of `(train command name, train command function)` tuples.
They can then be made available to the `sparkmodel` script via the `--custom-model` option.
For example, the `examples/custom_svm.py` file defines a custom SVM model, which can be trained via the command

    sparkmodel --custom-model examples/custom_svm.py train model data
 