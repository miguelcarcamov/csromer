#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""

from scipy.optimize import minimize
import numpy as np
from .fista import FISTA_algorithm
from .sdmm import sdmm
import proxmin as pmin
from abc import ABCMeta, abstractmethod
import sys


class Optimizer(metaclass=ABCMeta):

    def __init__(self, F_obj=None, i_guess=None, maxiter=None, method=None, tol=1e-15, verbose=True):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def setGuess(self, i_guess=None):
        self.i_guess = i_guess

    def setIter(self, maxiter=0):
        self.maxiter = maxiter

    def setTol(self, tol=1e-10):
        self.tol = tol

    @abstractmethod
    def run(self):
        return


class FixedPointMethod(Optimizer):
    def __init__(self, gx=None, **kwargs):
        super(FixedPointMethod, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def setFunction(self, gx=None):
        self.gx = gx

    def run(self):
        n = len(self.i_guess)
        xt = self.i_guess
        xt1 = np.zeros(n, dtype=self.i_guess.dtype)
        e = 1
        iter = 0

        while e > self.tol and iter < self.maxiter:
            xt1 = self.gx(xt)
            e = np.sum(np.abs(xt1 - xt))
            xt = xt1
            iter = iter + 1
        return xt1


class GradientBasedMethod(Optimizer):
    def __init__(self, method="CG", **kwargs):
        super(GradientBasedMethod, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        ret = minimize(fun=self.F_obj.evaluate, x0=self.i_guess, method=self.method, jac=self.F_obj.calculate_gradient,
                       tol=self.tol, options={'maxiter': self.maxiter, 'disp': self.verbose})
        return ret.fun, ret.x


class FISTA(Optimizer):
    def __init__(self, fx=None, gx=None, noise=None, n=None, **kwargs):
        super(FISTA, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        ret, x = FISTA_algorithm(self.i_guess, self.F_obj.evaluate, self.fx.calculate_gradient_fista, self.gx,
                                 self.maxiter, self.tol, self.n, self.noise, self.verbose)
        return ret, x


class ADMM(Optimizer):
    def __init__(self, fx=None, gx=None, L0=2, **kwargs):
        super(ADMM, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        x = self.i_guess
        converged, error = pmin.admm(x, prox_f=self.fx.calc_prox, step_f=None, prox_g=self.gx.calc_prox, L=None,
                                     e_rel=self.tol, max_iter=self.maxiter)
        # return
        # ret, x = FISTA_algorithm(self.i_guess, self.obj, self.fx, self.gx, self.gradfx, self.gprox,
        #               self.eta, self.maxiter, self.tol, self.verbose)
        return error, x


class SDMM(Optimizer):
    def __init__(self, fx=None, gx=None, gradfx=None, gprox=None, eta=2, **kwargs):
        super(SDMM, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        return
        # ret, x = FISTA_algorithm(self.i_guess, self.obj, self.fx, self.gx, self.gradfx, self.gprox,
        # self.eta, self.maxiter, self.tol, self.verbose)
        # converged, error =
        # return
