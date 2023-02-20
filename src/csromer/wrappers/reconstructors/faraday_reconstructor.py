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

    @staticmethod
    def estimate_peak_quadratic_interpolation(fd_signal, cellsize):
        length_n = len(fd_signal)
        index_0 = np.argmax(np.abs(fd_signal))

        fd_signal_0 = np.abs(fd_signal[index_0])
        fd_signal_m1 = np.abs(fd_signal[index_0 - 1])
        fd_signal_p1 = np.abs(fd_signal[index_0 + 1])

        pos_estimated_peak = (fd_signal_p1 - fd_signal_m1
                              ) / (4 * fd_signal_0 - 2 * fd_signal_m1 - 2 * fd_signal_p1)

        estimated_peak = (fd_signal_0 - 0.25 * (fd_signal_m1 - fd_signal_p1) * pos_estimated_peak)

        location = index_0 + pos_estimated_peak

        pos_phi_peak = (location - length_n / 2) * cellsize

        return pos_phi_peak, estimated_peak
