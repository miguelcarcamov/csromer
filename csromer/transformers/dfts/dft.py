from dataclasses import dataclass

import numpy as np

from .ft import FT


@dataclass(init=True, repr=True)
class DFT1D(FT):

    def __post_init__(self):
        super().__post_init__()

    def configure(self):
        return

    def forward(self, x):

        b = np.zeros(self.dataset.m, dtype=np.complex64)
        for i in range(0, self.dataset.m):
            b[i] = np.sum(x * np.exp(2.0j * self.parameter.phi * (self.dataset.lambda2[i])))

        return b

    def forward_normalized(self, x):

        # change units of x so the transform give us W(\lambda^2)*P(\lambda^2)
        val = x * self.k / self.parameter.n

        b = np.zeros(self.dataset.m, dtype=np.complex64)

        for i in range(0, self.dataset.m):
            b[i] = np.sum(val * np.exp(2.0j * self.parameter.phi * (self.dataset.lambda2[i])))

        notzero_idx = np.where(self.weights > 0.0)
        zero_idx = np.where(self.weights == 0.0)
        b[notzero_idx] /= self.weights[notzero_idx]
        b[zero_idx] = 0.0
        return b * self.dataset.s

    def backward(self, b):
        x = np.zeros(self.parameter.n, dtype=np.complex64)
        l2 = self.dataset.lambda2 - self.dataset.l2_ref
        for i in range(0, self.parameter.n):
            x[i] = np.sum(
                self.weights * b / self.dataset.s * np.exp(-2.0j * self.parameter.phi[i] * l2)
            )

        return x / self.k

    def RMTF(self, phi_x=0.0):
        x = np.zeros(self.parameter.n, dtype=np.complex64)
        l2 = self.dataset.lambda2 - self.dataset.l2_ref

        for i in range(0, self.parameter.n):
            x[i] = np.sum(self.dataset.w * np.exp(-2.0j * (self.parameter.phi[i] - phi_x) * l2))

        return x / self.dataset.k
