import numpy as np
from abc import ABCMeta, abstractmethod
import pywt
import sys


class Wavelet(metaclass=ABCMeta):
    def __init__(
        self,
        wavelet_name: str = None,
        level: int = None,
        mode: str = None,
        append_signal: bool = None,
    ):
        self.wavelet_name = wavelet_name
        self.mode = mode
        self.level = level
        self.append_signal = append_signal
        self.ncoeffs = 0
        self.n = 0
        self.wavelet = None
        self.coeff_slices = None

        if not isinstance(self.wavelet_name, str):
            raise TypeError("The wavelet name is not a string")

    @abstractmethod
    def calculate_max_level(self, x):
        return

    @abstractmethod
    def decompose(self, x):
        return

    @abstractmethod
    def decompose_complex(self, x):
        return

    @abstractmethod
    def reconstruct(self, input_coeffs):
        return

    @abstractmethod
    def reconstruct_complex(self, input_coeffs):
        return
