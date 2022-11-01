from dataclasses import dataclass

import pywt

from .wavelet import Wavelet


@dataclass(init=True, repr=True)
class ContinuousWavelet(Wavelet):

    def __init__(self):
        super().__post_init__()

        if self.wavelet_name is not None and self.wavelet_name in pywt.wavelist(kind="continuous"):
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        else:
            raise ValueError("The wavelet is not continuous")

    def calculate_max_level(self, x):
        return

    def decompose(self, x):
        return

    def decompose_complex(self, x):
        return

    def reconstruct(self, input_coeffs):
        return

    def reconstruct_complex(self, input_coeffs):
        return
