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
    nufft_instance: NUFFT = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.nufft_instance = NUFFT()

        if self.conv_size is None:
            self.conv_size = 4

        if self.oversampling_factor is None:
            self.oversampling_factor = 1

        if self.normalize is None:
            self.normalize = True

        if self.solve is None:
            self.solve = False

        if self.parameter.cellsize is not None:
            self.configure()

    def configure(self):
        l2_difference = self.dataset.lambda2 - self.dataset.l2_ref
        exp_factor = -2. * l2_difference * self.parameter.cellsize

        Nd = (len(self.parameter.phi), )  # Faraday Depth Space Length
        Kd = (
            self.oversampling_factor * len(self.parameter.phi),
        )  # Oversampled Faraday Depth Space Length
        Jd = (self.conv_size, )  # Convolution kernel size

        om_exp = np.reshape(exp_factor, (self.dataset.m, 1))  # Reshaping for nufft convention

        self.nufft_instance.plan(om_exp, Nd, Kd, Jd)

    def forward(self, x):
        b = self.nufft_instance.forward(x)
        return b

    def forward_normalized(self, x):
        val = x * self.dataset.k
        b = self.nufft_instance.forward(val)
        b = np.divide(b, self.dataset.w, where=self.dataset.w > 0.0)

        return b * self.dataset.s / len(self.parameter.phi)

    def backward(self, b, solver="cg", maxiter=1):
        weights = self.dataset.w / self.dataset.s
        if self.solve:
            x = self.nufft_instance.solve(weights * b, solver=solver, maxiter=maxiter)
        else:
            x = self.nufft_instance.adjoint(weights * b)

        if self.normalize:
            x *= len(self.parameter.phi) / self.dataset.k

        return x

    def RMTF(self):
        weights = self.dataset.w / self.dataset.s
        x = self.nufft_instance.adjoint(weights)
        x *= len(self.parameter.phi) / self.dataset.k
        return x
