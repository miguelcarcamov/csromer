from setuptools import setup, find_packages
import distutils.text_file
from typing import List

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(
        filename=str(Path(__file__).with_name(filename))).readlines()


setup(
    name="csromer",
    version="0.0.2",
    description="Compressed Sensing Rotation Measure Reconstructor",
    license="GNU GPL",
    url="https://github.com/miguelcarcamov/csromer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Miguel Carcamo",
    author_email="miguel.carcamo@manchester.ac.uk",
    packages=find_packages(),  # same as name
    install_requires=_parse_requirements(
        "requirements.txt"),  # external packages as dependencies
    scripts=[],
)
