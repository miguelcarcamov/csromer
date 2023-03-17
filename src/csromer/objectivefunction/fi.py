from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from ..dictionaries import Wavelet


@dataclass(init=True, repr=True)
class Fi(metaclass=ABCMeta):
    reg: float = None
    norm_factor: float = None
    wavelet: Wavelet = None

    def __post_init__(self):
        if self.reg is None:
            self.reg = 1.0

        if self.norm_factor is None:
            self.norm_factor = 1.0

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def calculate_gradient(self, x):
        pass

    @abstractmethod
    def calculate_prox(self, x, nu):
        pass
