from setuptools import setup

setup(
    name='main',
    version='1.0.0',
    entry_points="""
    [console_scripts]
    sparkmodel=sparkmodel.main:main
    """,
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    description='Minimal application to train models on Spark',
    install_requires=['click', 'pandas', 'scikit-learn']
)
