from setuptools import setup
import distutils.text_file
from typing import List
# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))
                                        ).readlines()


if __name__ == "__main__":
    use_scm_version = {"root": ".", "relative_to": __file__, "local_scheme": "node-and-timestamp"}
    setup(use_scm_version=use_scm_version, install_requires=_parse_requirements("requirements.txt"))
