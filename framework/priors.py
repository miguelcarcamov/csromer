#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:38:54 2019

@author: miguel
"""
import numpy as np
import prox_tv as ptv
from .utilities import real_to_complex, complex_to_real
from abc import ABCMeta, abstractmethod
from .optimizer import FixedPointMethod, GradientBasedMethod
import pywt
import sys

def approx_abs(x, epsilon):
    return np.sqrt(x * x + epsilon)


class Fi(metaclass=ABCMeta):
    def __init__(self, reg=1.0, norm_factor=1.0, wavelet=None):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    @abstractmethod
    def evaluate(self, x):
        return

    @abstractmethod
    def calculate_gradient(self, x):
        return

    @abstractmethod
    def calculate_prox(self, x, nu):
        return


class TV(Fi):
    def __init__(self, **kwargs):
        super(TV, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

        self.nu = np.array([])

    def evaluate(self, x):
        tv = 0.0
        n = x.shape[0]
        for i in range(0, n - 1):
            tv += np.abs(x[i + 1] - x[i])
        return tv

    def calculate_gradient(self, x):

        n = len(x)
        dx = np.zeros(n, x.dtype)
        for i in range(1, n - 1):
            dx[i] = np.sign(x[i] - x[i - 1]) - np.sign(x[i + 1] - x[i])
        return dx

    def calculate_prox(self, x, nu=0.0):
        return ptv.tv1_1d(x, self.reg)


class TSV(Fi):
    def __init__(self, **kwargs):
        super(TSV, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

        self.nu = np.array([])

    def evaluate(self, x):
        tv = 0.0
        n = x.shape[0]
        for i in range(0, n - 1):
            tv += np.abs(x[i + 1] - x[i])**2
        return tv

    def calculate_gradient(self, x):

        n = len(x)
        dx = np.zeros(n, x.dtype)
        for i in range(1, n - 1):
            dx[i] = 2.0 * (np.sign(x[i] - x[i - 1]) - np.sign(x[i + 1] - x[i]))
        return dx

    def calculate_prox(self, x, nu=0.0):
        return ptv.tv2_1d(x, self.reg)


class L1(Fi):
    def __init__(self, **kwargs):
        super(L1, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def evaluate(self, x, epsilon=1e-12):
        val = np.sum(approx_abs(x, epsilon))
        #print("Evaluation on L1:", val)
        return val

    def calculate_gradient(self, x, epsilon=1e-12):
        dx = np.zeros(len(x), x.dtype)

        dx = x / approx_abs(x, epsilon)

        return dx

    def calculate_prox(self, x, nu=0):
        # print(x)
        x = np.sign(x) * np.maximum(np.abs(x) - self.reg, 0.0)
        return x


class Chi2(Fi):
    def __init__(self, dft_obj=None, **kwargs):
        super(Chi2, self).__init__(**kwargs)
        self.dft_obj = dft_obj
        self.F_dirty = None
        if self.dft_obj is not None:
            self.F_dirty = self.dft_obj.backward(self.dft_obj.dataset.data)

    def evaluate(self, x):
        if self.wavelet is not None:
            x = self.wavelet.reconstruct(x)
        x_complex = real_to_complex(x) * self.norm_factor
        model_data = self.dft_obj.forward_normalized(x_complex)
        self.dft_obj.dataset.model_data = model_data
        # res = model_data - self.dft_obj.dataset.data
        res = self.dft_obj.dataset.residual
        chi2_vector = self.dft_obj.dataset.w * (res.real ** 2 + res.imag ** 2)
        val = 0.5 * np.sum(chi2_vector)
        # print("Evaluation on chi2:", val)
        return val

    def calculate_gradient(self, x):
        if self.wavelet is not None:
            x = self.wavelet.reconstruct(x)
        x_complex = real_to_complex(x) * self.norm_factor
        val = x_complex - self.F_dirty
        return complex_to_real(val)

    def calculate_gradient_fista(self, x):
        if self.wavelet is not None:
            x = self.wavelet.reconstruct(x)
        x_complex = real_to_complex(x) * self.norm_factor
        model_data = self.dft_obj.forward_normalized(x_complex)
        self.dft_obj.dataset.model_data = model_data
        # res = model_data - self.dft_obj.dataset.data
        res = self.dft_obj.dataset.residual
        val = self.dft_obj.backward(res)
        ret_val = complex_to_real(val)
        if self.wavelet is not None:
            ret_val = self.wavelet.decompose(ret_val)
        return ret_val

    def calculate_prox(self, x, nu=0):
        a_transpose_b = complex_to_real(self.F_dirty)
        lambda_plus_one = self.reg + 1.0
        one_over_lambda1 = 1.0 / lambda_plus_one
        return one_over_lambda1 * (self.reg * a_transpose_b + nu)
