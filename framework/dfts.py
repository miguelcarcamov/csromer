#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from pynufft import NUFFT


class FT(metaclass=ABCMeta):
    def __init__(self, W=None, lambda2=None, lambda2_ref=0.0, phi=None):
        self.lambda2 = lambda2
        self.phi = phi
        self.lambda2_ref = lambda2_ref
        self.m = len(lambda2)
        self.n = len(phi)
        self.W = W
        if self.W is None:
            self.W = np.ones_like(lambda2)

        self.K = np.sum(self.W)

    @property
    def lambda2(self):
        return self.__lambda2

    @lambda2.setter
    def lambda2(self, val):
        self.__lambda2 = val
        self.__m = len(val)

    @property
    def phi(self):
        return self.__phi

    @phi.setter
    def phi(self, val):
        self.__phi = val
        self.__n = len(val)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, val):
        self.__W = val
        self.__K = np.sum(val)

    @abstractmethod
    def configure(self):
        return

    @abstractmethod
    def forward(self):
        return

    @abstractmethod
    def forward_normalized(self):
        return

    @abstractmethod
    def backward(self):
        return


class DFT1D(FT):
    def __init__(self, **kwargs):
        super(DFT1D, self).__init__(**kwargs)

    def configure(self):
        return

    def forward(self, x):

        b = np.zeros(self.m, dtype=np.float32) + 1j * np.zeros(self.m, dtype=np.float32)

        for i in range(0, self.m):
            b[i] = np.sum(x * np.exp(2 * 1j * self.phi * (self.lambda2[i] - self.lambda2_ref)))

        return self.W * b

    def forward_normalized(self, x):

        # change units of x so the transform give us W(\lambda^2)*P(\lambda^2)
        val = x * self.K / self.n

        b = np.zeros(self.m, dtype=np.float32) + 1j * np.zeros(self.m, dtype=np.float32)

        for i in range(0, self.m):
            b[i] = np.sum(val * np.exp(2 * 1j * self.phi * (self.lambda2[i] - self.lambda2_ref)))

        notzero_idx = np.where(self.W != 0.0)
        zero_idx = np.where(self.W == 0.0)
        b[notzero_idx] /= self.W[notzero_idx]
        b[zero_idx] = 0.0
        return b

    def backward(self, b):
        x = np.zeros(self.n, dtype=np.float32) + 1j * np.zeros(self.n, dtype=np.float32)
        l2 = self.lambda2 - self.lambda2_ref
        for i in range(0, self.n):
            x[i] = np.sum(self.W * b * np.exp(-2 * 1j * self.phi[i] * l2))

        return (1. / self.K) * x

    def RMTF(self, phi_x=0.0):
        x = np.zeros(self.n, dtype=np.float32) + 1j * np.zeros(self.n, dtype=np.float32)
        l2 = self.lambda2 - self.lambda2_ref
        for i in range(0, self.n):
            x[i] = np.sum(self.W * np.exp(-2 * 1j * (self.phi[i] - phi_x) * l2))

        return (1. / self.K) * x


class NUFFT1D(FT):
    def __init__(self, conv_size=4, oversampling_factor=1, normalize=True, solve=True, **kwargs):
        super(NUFFT1D, self).__init__(**kwargs)
        self.nufft_obj = NUFFT()
        self.conv_size = conv_size
        self.oversampling_factor = oversampling_factor
        self.normalize = normalize
        self.solve = solve
        if self.lambda2 is not None and self.phi is not None:
            self.delta_phi = np.abs(self.phi[1] - self.phi[0])
            self.configure()

    def configure(self):
        exp_factor = -2.0 * (self.lambda2 - self.lambda2_ref) * self.delta_phi

        Nd = (self.n,)  # Faraday Depth Space Length
        Kd = (self.oversampling_factor * self.n,)  # Oversampled Faraday Depth Space Length
        Jd = (self.conv_size,)
        om = np.reshape(exp_factor, (self.m, 1))  # Exponential data
        self.nufft_obj.plan(om, Nd, Kd, Jd)

    def forward(self, x):
        b = self.nufft_obj.forward(x)
        return self.W * b

    def forward_normalized(self, x):
        val = x * self.K / self.n
        b = self.nufft_obj.forward(val)

        notzero_idx = np.where(self.W != 0.0)
        zero_idx = np.where(self.W == 0.0)
        b[notzero_idx] /= self.W[notzero_idx]
        b[zero_idx] = 0.0
        return b

    def backward(self, b, solver="cg", maxiter=100):
        if self.solve:
            x = self.nufft_obj.solve(b, solver=solver, maxiter=maxiter)
        else:
            x = self.nufft_obj.adjoint(b)

        if self.normalize:
            x *= self.n / self.K
        return x
