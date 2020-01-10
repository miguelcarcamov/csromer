#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""

from scipy.optimize import minimize
from fista import FISTA


class Optimizer:
    maxiter = 500
    tol = 1e-12

    def __init__(self, obj=None, grad=None, i_guess=None, maxiter=500, method=None, tol=1e-15, verbose=True):
        self.obj = obj
        self.grad = grad
        self.i_guess = i_guess
        self.maxiter = maxiter
        self.method = method
        self.tol = tol
        self.verbose = verbose

    def gradient_based_method(self):
        ret = minimize(fun=self.obj, x0=self.i_guess, method=self.method, jac=self.grad,
                       tol=self.tol, options={'maxiter': self.maxiter, 'disp': self.verbose})
        return ret

    def FISTA(self, fx, gx, gradfx, prox, eta):
        ret, x = FISTA(self.i_guess, self.obj, fx, gx, gradfx,
                       prox, eta, self.maxiter, self.tol, self.verbose)
        return ret, x
