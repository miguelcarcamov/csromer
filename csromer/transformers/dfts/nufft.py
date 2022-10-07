from dataclasses import dataclass, field

import numpy as np
from pynufft import NUFFT

from .ft import FT


@dataclass(init=True, repr=True)
class NUFFT1D(FT):
    conv_size: int = None
    oversampling_factor: int = None
    normalize: bool = None
    solve: bool = None
    nufft_forward: NUFFT = field(init=False)
    nufft_backward: NUFFT = field(init=False)
    delta_phi: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.nufft_forward = NUFFT()
        self.nufft_backward = NUFFT()

        if self.conv_size is None:
            self.conv_size = 4

        if self.oversampling_factor is None:
            self.oversampling_factor = 1

        if self.normalize is None:
            self.normalize = True

        if self.solve is None:
            self.solve = False

        if self.parameter.cellsize is not None:
            self.delta_phi = self.parameter.cellsize
            self.configure()

    def configure(self):
        l2_forward = self.dataset.lambda2 - self.dataset.l2_ref
        l2_backward = self.dataset.lambda2 - self.dataset.l2_ref
        exp_factor_forward = -2. * l2_forward * self.parameter.cellsize
        exp_factor_backward = -2. * l2_backward * self.parameter.cellsize

        Nd = (self.parameter.n, )  # Faraday Depth Space Length
        Kd = (
            self.oversampling_factor * self.parameter.n,
        )  # Oversampled Faraday Depth Space Length
        Jd = (self.conv_size, )  # Convolution kernel size

        om_forward = np.reshape(
            exp_factor_forward, (self.dataset.m, 1)
        )  # Exponential data backward transform
        om_backward = np.reshape(
            exp_factor_backward, (self.dataset.m, 1)
        )  # Exponential data backward transform

        self.nufft_forward.plan(om_forward, Nd, Kd, Jd)
        self.nufft_backward.plan(om_backward, Nd, Kd, Jd)

    def forward(self, x):
        b = self.nufft_forward.forward(x)
        return b

    def forward_normalized(self, x):
        val = x * self.k / self.parameter.n
        b = self.nufft_forward.forward(val)
        b *= self.dataset.s
        notzero_idx = np.where(self.dataset.w > 0.0)
        zero_idx = np.where(self.dataset.w == 0.0)
        b[notzero_idx] /= self.weights[notzero_idx]
        b[zero_idx] = 0.0
        return b

    def backward(self, b, solver="cg", maxiter=100000):
        if self.solve:
            x = self.nufft_backward.solve(
                self.weights * b / self.dataset.s, solver=solver, maxiter=maxiter
            )
        else:
            x = self.nufft_backward.adjoint(self.weights * b / self.dataset.s)

        if self.normalize:
            x *= self.parameter.n / self.k

        return x

    def RMTF(self):
        x = self.nufft_backward.adjoint(self.dataset.w)
        return x / self.dataset.k
