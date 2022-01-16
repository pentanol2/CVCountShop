
import os
from setuptools import setup

setup(
    name="Instance Counter Shop",
    version="1.0.0",
    description="Object Tracker for instance counting and more",
    long_description=open(os.path.abspath(os.path.dirname(__file__)) ,"README.md"),
    long_description_content_type="text/markdown",
    author="YOUSSEF AIDANI",
    install_requires=["setuptools"]
)