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
        for i in range(0,n-1):
                tv += np.abs(x[i+1] - x[i])
        return tv

    def calculate_gradient(self, x):

         n = len(x)
         dx = np.zeros(n, x.dtype)
         for i in range(1,n-1):
                 dx[i] = np.sign(x[i]-x[i-1]) - np.sign(x[i+1]-x[i])
         return dx

class L1:
    reg = 0.0
    def __init__(self, reg):
        self.reg = reg

    def evaluate(self, x):
        return np.linalg.norm(x,ord=1)

    def calculate_gradient(self, x):
        dx = np.zeros(len(x), x.dtype)
        idx = np.argwhere(x != 0)
        idx_0 = np.argwhere(x==0)

        dx[idx] = np.sign(x[idx])
        dx[idx_0] = 0.0
        return dx

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
