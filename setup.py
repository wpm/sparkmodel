from setuptools import setup
from sparkmodel import __version__

setup(
    name="Sparkmodel",
    version=__version__,
    entry_points="""
    [console_scripts]
    sparkmodel=sparkmodel.main:main
    """,
    scripts=["bin/sparkmodel_submit.py"],
    python_requires=">=3.6",
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Minimal application to train models on Spark",
    install_requires=["click", "pyspark", "sklearn", "numpy"]
)
