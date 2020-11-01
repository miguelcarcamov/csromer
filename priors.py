#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:38:54 2019

@author: miguel
"""
import numpy as np
from utilities import real_to_complex, complex_to_real


class TV:
    reg = 0.0

    def __init__(self, reg):
        self.reg = reg

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

    def calculate_prox(self, x, nu):
        # This derivative is calculated using fixed point method

        if self.reg != 0.0:
            n = len(x)
            xt1 = np.zeros(n, dtype=x.dtype)
            xt = x
            maxiter = 500
            tol = 1e-6
            e = 1
            iter = 0
            while e > tol and iter < maxiter:
                for i in range(1, n - 1):
                    xt1[i] = nu[i] - self.reg * \
                        (np.sign(xt[i] - xt[i - 1]) -
                         np.sign(xt[i + 1] - xt[i]))
                e = np.linalg.norm(xt - xt1)
                xt = xt1
                #print("Iter: ", iter, " - Norm: ", e)
                iter = iter + 1
        else:
            xt1 = nu

        return xt1


class L1:
    reg = 0.0

    def __init__(self, reg):
        self.reg = reg

    def evaluate(self, x):
        return np.linalg.norm(x, ord=1)

    def calculate_gradient(self, x):
        dx = np.zeros(len(x), x.dtype)
        idx = np.argwhere(x != 0)
        idx_0 = np.argwhere(x == 0)

        dx[idx] = np.sign(x[idx])
        dx[idx_0] = 0.0
        return dx

    def calculate_prox(self, x, nu=0):
        # print(x)
        x = np.sign(x) * np.maximum(0, np.abs(x) - self.reg)
        return x


class chi2:
    reg = 1.0

    def __init__(self, b, dft_obj, w=1.0, reg=1.0):
        self.b = b
        self.dft = dft_obj
        self.F_dirty = self.dft.backward(self.b)
        self.reg = reg
        self.w = w

    def evaluate(self, x):
        x_complex = real_to_complex(x)

        res = self.dft.forward(x_complex) - (self.w * self.b)

        return 0.5 * np.sum((res.real**2) + (res.imag**2))

    def calculate_gradient(self, x):
        res = x - complex_to_real(self.F_dirty)

        return res

    def calculate_prox(self, x, nu=0):
        a_transpose_b = complex_to_real(self.F_dirty)
        lambda_plus_one = self.reg + 1.0
        one_over_lambda1 = 1.0 / lambda_plus_one
        return one_over_lambda1 * (self.reg * a_transpose_b + nu)
