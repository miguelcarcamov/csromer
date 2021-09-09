import numpy as np
from abc import ABCMeta, abstractmethod
import pywt
import sys


class Wavelet(metaclass=ABCMeta):
    def __init__(self, wavelet_name: str = None, level: int = None, mode: str = None):
        self.wavelet_name = wavelet_name
        self.mode = mode
        self.level = level
        self.ncoeffs = 0
        self.wavelet = None
        self.coeff_slices = None

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
