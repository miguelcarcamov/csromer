from dataclasses import dataclass

import numpy as np

from .ft import FT


@dataclass(init=True, repr=True)
class NDFT1D(FT):

    def __post_init__(self):
        super().__post_init__()

    def configure(self):
        return

    def forward(self, x):
        l2 = self.dataset.lambda2[np.newaxis, :] - self.dataset.l2_ref
        return np.dot(x, np.exp(2.0j * l2 * self.parameter.phi[:, np.newaxis])).astype(np.complex64)

    def forward_normalized(self, x):
        l2 = self.dataset.lambda2[np.newaxis, :] - self.dataset.l2_ref
        # change units of x so the transform give us W(\lambda^2)*P(\lambda^2)
        val = x * self.k
        b = np.dot(val, np.exp(2.0j * l2 * self.parameter.phi[:, np.newaxis])).astype(np.complex64)
        b = np.divide(b, self.dataset.w, where=self.dataset.w > 0.)

        return b * self.dataset.s / len(self.parameter.phi)

    def backward(self, b):
        l2 = self.dataset.lambda2[:, np.newaxis] - self.dataset.l2_ref
        weights = self.dataset.w / self.dataset.s
        x = np.dot(weights * b,
                   np.exp(-2.0j * l2 * self.parameter.phi[np.newaxis, :])).astype(np.complex64)

        return x / self.dataset.k

    def RMTF(self, phi_x=0.0):
        l2 = self.dataset.lambda2[:, np.newaxis] - self.dataset.l2_ref
        weights = self.dataset.w / self.dataset.s
        x = np.dot(weights,
                   np.exp(-2.0j * l2 * self.parameter.phi[np.newaxis, :])).astype(np.complex64)
        return x / self.dataset.k
