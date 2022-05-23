from setuptools import setup, find_packages
from handyrec import __version__

REQUIRES = ["numpy>=1.21.5", "pandas>=1.3.4", "python_box>=6.0.2", "tensorflow>=2.6.0"]

setup(
    name="handyrec",
    version=__version__,
    author="Weipeng Zhang",
    author_email="wp.zhang@hotmail.com",
    packages=find_packages(
        exclude=[
            "tests",
            "tests.models",
            "tests.layers",
            "tests.data",
            "tests.features",
            "tests.models.ranking",
            "tests.models.retrieval",
        ]
    ),
    install_requires=REQUIRES,
)
