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
        self.m = len(lambda2)
        self.n = len(phi)

    def forward(self, x):
        # change units of x so the transform give us W(\lambda^2)*P(\lambda^2)
        # val = x*self.K / self.n

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

    def RMTF(self):
        x = np.zeros(self.n, dtype=np.float32) + 1j * np.zeros(self.n, dtype=np.float32)
        l2 = self.lambda2 - self.lambda2_ref
        for i in range(0, self.n):
            x[i] = np.sum(self.W * np.exp(-2 * 1j * self.phi[i] * l2))

        return (1. / self.K) * x
