import numpy as np
from pynufft import NUFFT

from .ft import FT


class NUFFT1D(FT):

    def __init__(self, conv_size=4, oversampling_factor=1, normalize=True, solve=True, **kwargs):
        super().__init__(**kwargs)
        self.nufft_obj = NUFFT()
        self.conv_size = conv_size
        self.oversampling_factor = oversampling_factor
        self.normalize = normalize
        self.solve = solve
        if self.parameter.cellsize is not None:
            self.delta_phi = self.parameter.cellsize
            self.configure()

    def configure(self):
        l2 = self.dataset.lambda2 - self.dataset.l2_ref
        exp_factor = -2.0 * l2 * self.parameter.cellsize

        Nd = (self.parameter.n, )  # Faraday Depth Space Length
        Kd = (
            self.oversampling_factor * self.parameter.n,
        )  # Oversampled Faraday Depth Space Length
        Jd = (self.conv_size, )
        om = np.reshape(exp_factor, (self.dataset.m, 1))  # Exponential data
        self.nufft_obj.plan(om, Nd, Kd, Jd)

    def forward(self, x):
        b = self.nufft_obj.forward(x)
        return b

    def forward_normalized(self, x):
        val = x * self.k / self.parameter.n
        b = self.nufft_obj.forward(val)
        b *= self.dataset.s
        notzero_idx = np.where(self.dataset.w > 0.0)
        zero_idx = np.where(self.dataset.w == 0.0)
        b[notzero_idx] /= self.weights[notzero_idx]
        b[zero_idx] = 0.0
        return b

    def backward(self, b, solver="cg", maxiter=1):
        if self.solve:
            x = self.nufft_obj.solve(
                self.weights * b / self.dataset.s, solver=solver, maxiter=maxiter
            )
        else:
            x = self.nufft_obj.adjoint(self.weights * b / self.dataset.s)

        if self.normalize:
            x *= self.parameter.n / self.k
            # x *= self.parameter.n / len(self.dataset.w)
        return x

    def RMTF(self):
        x = self.nufft_obj.adjoint(self.dataset.w)
        return x / self.dataset.k
