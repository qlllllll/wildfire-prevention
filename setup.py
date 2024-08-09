import os

from setuptools import setup

package = ["fire_risk_object_detection"]
version = "0.0.1"
description = "A collection of utility functions for processing Google Streetview Images along with geospatial data"
author = "Qijun Li"
author_email = ""

if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()
else:
    long_description = description

install_requires = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as f:
        install_requires = f.read().splitlines()

setup(
    name="fire_risk_object_detection",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    packages=package,
    install_requires=install_requires,
    python_requires=">=3.11",
)