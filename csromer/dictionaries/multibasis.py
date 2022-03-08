from .wavelet import Wavelet
import pywt
import numpy as np


class MultiBasis(Wavelet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_max_level(self, x):


    def decompose(self, x):


    def decompose_complex(self, x):


    def reconstruct(self, input_coeffs):


    def reconstruct_complex(self, input_coeffs):
