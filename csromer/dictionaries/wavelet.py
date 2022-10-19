from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from pywt import Wavelet


@dataclass(init=True, repr=True)
class Wavelet(metaclass=ABCMeta):
    wavelet_name: str = None
    wavelet_level: int = None
    mode: str = None
    append_signal: bool = None
    ncoeffs: int = field(init=False, default=0)
    n: int = field(init=False, default=0)
    wavelet: Wavelet = field(init=False, default=None)
    coeff_slices: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
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
