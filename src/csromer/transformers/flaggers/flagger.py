from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np

from csromer.base import Dataset


def median_absolute_deviation(x):
    """
    Returns the median absolute deviation from the window's median
    :param x: Values in the window
    :return: MAD
    """
    return np.median(np.abs(x - np.median(x)))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


@dataclass(init=True, repr=True)
class Flagger(metaclass=ABCMeta):

    dataset: Union[Dataset] = None
    nsigma: float = None
    delete_channels: bool = None

    def __post_init__(self):
        if self.nsigma is None:
            self.sigma = 0.0

        if self.delete_channels is None:
            self.delete_channels = False

    @abstractmethod
    def run(self):
        pass
