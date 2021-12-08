from setuptools import setup
import distutils.text_file
from pathlib import Path
from typing import List


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()


with open("README.md", 'r') as f:
    long_description = f.read()

    setup(
        name='csromer',
        version='0.0.1',
        description='Compressed Sensing Rotation Measure Reconstructor',
        license="GNU GPL",
        long_description=long_description,
        author='Miguel Carcamo',
        author_email='miguel.carcamo@manchester.ac.uk',
        packages=['csromer'],  # same as name
        install_requires=_parse_requirements('requirements.txt'),  # external packages as dependencies
        scripts=[]
    )
