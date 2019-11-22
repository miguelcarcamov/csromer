#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:46:13 2019

@author: miguel
"""
import numpy as np

class soft_thresholding:
    reg = 0.0
    def __init__(self, reg = 0.0):
        self.reg = reg

    def setreg(self, reg):
        self.reg = reg

    def calculate(self, x):
        x = np.sign(x)*np.maximum(0, np.abs(x) - self.reg)
        return x

class hard_thresholding:
    reg = 0.0
    def __init__(self, reg = 0.0):
        self.reg = reg

    def setreg(self, reg):
        self.reg = reg

    def calculate(self, x):
        j = np.abs(x) < self.reg
        x[j] = 0
        return x

class total_variation:
    reg = 0.0
    def __init__(self, reg = 0.0):
        self.reg = reg

    def setreg(self, reg):
        self.reg = reg

    def calculate(self, x, nu):
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
                for i in range(1, n-1):
                    xt1[i] = nu[i] - self.reg*(np.sign(xt[i]-xt[i-1]) - np.sign(xt[i+1]-xt[i]))
                e = np.linalg.norm(xt-xt1)
                xt = xt1
                print("Iter: ", iter, " - Norm: ", e)
                iter = iter + 1
        else:
            xt1 = nu
        return xt1
