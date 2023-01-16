import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dependencies = [
    "tqdm",
    "ipykernel",
    "trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3",
    "networkx",
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "tables",
    "wandb"
]

setup(
    name="gnn4na",
    version="0.0.1",
    description="A simple template",
    author="Daniel Murnane",
    install_requires=dependencies,
    packages=find_packages(include=["LightningModules", "LightningModules.*", "Scripts", "Scripts.*", "Utils", "Utils.*"]),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords=[
        "ATLAS",
        "track reconstruction",
        "graph networks",
        "machine learning",
    ],
)
