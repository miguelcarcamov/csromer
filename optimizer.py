#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""

from scipy.optimize import minimize

class Optimizer:
    maxiter = 500
    tol = 1e-12
    def __init__(self, obj=None, grad=None, i_guess=None, maxiter=500, method='CG', tol=1e-15, verbose=True):
        self.obj = obj
        self.grad = grad
        self.i_guess = i_guess
        self.maxiter=maxiter
        self.method=method
        self.tol=tol
        self.verbose=verbose
    
    def optimize(self):
        ret = minimize(fun=self.obj, x0=self.i_guess, method=self.method, jac=self.grad, tol=self.tol, options={'maxiter':self.maxiter, 'disp':self.verbose})
        return ret