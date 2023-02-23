from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...base import Dataset


@dataclass(init=True, repr=True)
class FaradayReconstructorWrapper(metaclass=ABCMeta):
    dataset: Dataset = None

    @abstractmethod
    def config_fd_space(self):
        pass

    @abstractmethod
    def reconstruct(self):
        pass

    @abstractmethod
    def calculate_second_moment(self):
        pass
