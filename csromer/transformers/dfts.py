#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""
from __future__ import annotations
import numpy as np
from abc import ABCMeta, abstractmethod
from pynufft import NUFFT
from ..reconstruction import Parameter
from scipy.constants import speed_of_light as c
from typing import TYPE_CHECKING
import copy
if TYPE_CHECKING:
    from ..base import Dataset


class FT(metaclass=ABCMeta):
    def __init__(self, dataset: Dataset = None, parameter: Parameter = None):
        self.dataset = dataset
        self.s = None
        if parameter is not None:
            self.parameter = copy.deepcopy(parameter)
        else:
            self.parameter = None

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

    @abstractmethod
    def RMTF(self):
        return


class DFT1D(FT):
    def __init__(self, **kwargs):
        super(DFT1D, self).__init__(**kwargs)

    def configure(self):
        return

    def forward(self, x):

        b = np.zeros(self.dataset.m, dtype=np.float32) + 1j * np.zeros(self.dataset.m, dtype=np.float32)
        for i in range(0, self.dataset.m):
            b[i] = np.sum(
                x * np.exp(2 * 1j * self.parameter.phi * (self.dataset.lambda2[i] - self.dataset.l2_ref)))

        return self.dataset.w * b * self.dataset.s

    def forward_normalized(self, x):

        # change units of x so the transform give us W(\lambda^2)*P(\lambda^2)
        val = x * self.dataset.k / self.parameter.n

        b = np.zeros(self.dataset.m, dtype=np.float32) + 1j * np.zeros(self.dataset.m, dtype=np.float32)

        for i in range(0, self.dataset.m):
            b[i] = np.sum(
                val * np.exp(2 * 1j * self.parameter.phi * (self.dataset.lambda2[i] - self.dataset.l2_ref)))

        notzero_idx = np.where(self.dataset.w != 0.0)
        zero_idx = np.where(self.dataset.w == 0.0)
        b[notzero_idx] /= self.dataset.w[notzero_idx]
        b[zero_idx] = 0.0
        return b * self.dataset.s

    def backward(self, b):
        x = np.zeros(self.parameter.n, dtype=np.float32) + 1j * np.zeros(self.parameter.n, dtype=np.float32)
        l2 = self.dataset.lambda2 - self.dataset.l2_ref
        for i in range(0, self.parameter.n):
            x[i] = np.sum(self.dataset.w * b / self.dataset.s * np.exp(-2 * 1j * self.parameter.phi[i] * l2))

        return x / self.dataset.k

    def RMTF(self, phi_x=0.0):
        x = np.zeros(self.parameter.n, dtype=np.float32) + 1j * np.zeros(self.parameter.n, dtype=np.float32)
        l2 = self.dataset.lambda2 - self.dataset.lambda2_ref
        for i in range(0, self.parameter.n):
            x[i] = np.sum(self.dataset.w * np.exp(-2 * 1j * (self.parameter.phi[i] - phi_x) * l2))

        return x / self.dataset.k


class NUFFT1D(FT):
    def __init__(self, conv_size=4, oversampling_factor=1, normalize=True, solve=True, **kwargs):
        super(NUFFT1D, self).__init__(**kwargs)
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

        Nd = (self.parameter.n,)  # Faraday Depth Space Length
        Kd = (self.oversampling_factor * self.parameter.n,)  # Oversampled Faraday Depth Space Length
        Jd = (self.conv_size,)
        om = np.reshape(exp_factor, (self.dataset.m, 1))  # Exponential data
        self.nufft_obj.plan(om, Nd, Kd, Jd)

    def forward(self, x):
        b = self.nufft_obj.forward(x)
        b *= self.dataset.s
        return self.dataset.w * b

    def forward_normalized(self, x):
        val = x * self.dataset.k / self.parameter.n
        b = self.nufft_obj.forward(val)
        b *= self.dataset.s
        notzero_idx = np.where(self.dataset.w != 0.0)
        zero_idx = np.where(self.dataset.w == 0.0)
        b[notzero_idx] /= self.dataset.w[notzero_idx]
        b[zero_idx] = 0.0
        return b

    def backward(self, b, solver="cg", maxiter=1):
        if self.solve:
            x = self.nufft_obj.solve(self.dataset.w * b / self.dataset.s, solver=solver, maxiter=maxiter)
        else:
            x = self.nufft_obj.adjoint(self.dataset.w * b / self.dataset.s)

        if self.normalize:
            x *= self.parameter.n / self.dataset.k
        return x

    def RMTF(self):
        x = self.nufft_obj.adjoint(self.dataset.w)
        return (1. / self.dataset.k) * x
