#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""

import numpy as np


class DFT1D:
    def __init__(self, W, K, lambda2, lambda2_ref, phi):
        self.W = W
        self.K = K
        self.lambda2 = lambda2
        self.lambda2_ref = lambda2_ref
        self.phi = phi

    def forward(self, x):
        m = len(self.lambda2)
        b = np.zeros(m) + 1j * np.zeros(m)
        for i in range(0, m):
            b[i] = np.sum(x * np.exp(2j * self.phi *
                                     (self.lambda2[i] - self.lambda2_ref)))
        return self.W * b

    def backward(self, b):
        n = len(self.phi)
        x = np.zeros(n) + 1j * np.zeros(n)

        for i in range(0, n):
            x[i] = np.sum(self.W * b * np.exp(-2j * self.phi[i]
                                              * (self.lambda2 - self.lambda2_ref)))

        return self.K * x

    def backward_normalized(self, b):
        n = len(self.phi)
        x = np.zeros(n) + 1j * np.zeros(n)

        for i in range(0, n):
            x[i] = np.sum(self.W * b * np.exp(-2j * self.phi[i]
                                              * (self.lambda2 - self.lambda2_ref)))

        return x / n
