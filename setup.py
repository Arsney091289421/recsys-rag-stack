# setup.py
from setuptools import setup, find_packages

setup(
    name="recys_rag_stack",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
