from setuptools import setup
from sparkmodel import __version__

setup(
    name='Sparkmodel',
    version=__version__,
    entry_points="""
    [console_scripts]
    sparkmodel=sparkmodel.main:main
    """,
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    description='Minimal application to train models on Spark',
    install_requires=['click', 'pandas', 'scikit-learn']
)
