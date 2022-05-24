#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""
import sys
import copy
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize
import numpy as np
import proxmin as pmin
from .methods.fista import FISTA_algorithm
from .methods.sdmm import sdmm
from ..reconstruction.parameter import Parameter


class Optimizer(metaclass=ABCMeta):

    def __init__(
        self,
        guess_param: Parameter = None,
        F_obj=None,
        maxiter=None,
        method=None,
        tol=np.finfo(np.float32).tiny,
        verbose=True,
    ):
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    @abstractmethod
    def run(self):
        return


class FixedPointMethod(Optimizer):

    def __init__(self, gx=None, **kwargs):
        super(FixedPointMethod, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        n = self.guess_param.n
        xt = self.guess_param.data
        xt1 = np.zeros(n, dtype=xt.dtype)
        e = 1
        iter = 0

        while e > self.tol and iter < self.maxiter:
            xt1 = self.gx(xt)
            e = np.sum(np.abs(xt1 - xt))
            xt = xt1
            iter = iter + 1

        param = copy.deepcopy(self.guess_param)
        param.data = xt1
        return e, param


class GradientBasedMethod(Optimizer):

    def __init__(self, method="CG", **kwargs):
        super(GradientBasedMethod, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        ret = minimize(
            fun=self.F_obj.evaluate,
            x0=self.guess_param.data,
            method=self.method,
            jac=self.F_obj.calculate_gradient,
            tol=self.tol,
            options={
                "maxiter": self.maxiter,
                "disp": self.verbose
            },
        )

        param = copy.deepcopy(self.guess_param)
        param.data = ret.x
        return ret.fun, param


class FISTA(Optimizer):

    def __init__(self, fx=None, gx=None, noise=None, **kwargs):
        super(FISTA, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        ret, x = FISTA_algorithm(
            self.guess_param.data,
            self.F_obj.evaluate,
            self.fx.calculate_gradient_fista,
            self.gx,
            self.maxiter,
            self.tol,
            self.guess_param.n,
            self.noise,
            self.verbose,
        )

        param = copy.deepcopy(self.guess_param)
        param.data = x
        return ret, param


class ADMM(Optimizer):

    def __init__(self, fx=None, gx=None, L0=2, **kwargs):
        super(ADMM, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        x = self.guess_param.data
        converged, error = pmin.admm(
            x,
            prox_f=self.fx.calc_prox,
            step_f=None,
            prox_g=self.gx.calc_prox,
            L=None,
            e_rel=self.tol,
            max_iter=self.maxiter,
        )
        # return
        # ret, x = FISTA_algorithm(self.i_guess, self.obj, self.fx, self.gx, self.gradfx, self.gprox,
        #               self.eta, self.maxiter, self.tol, self.verbose)
        param = copy.deepcopy(self.guess_param)
        param.data = x
        return error, param


class SDMM(Optimizer):

    def __init__(self, fx=None, gx=None, gradfx=None, gprox=None, eta=2, **kwargs):
        super(SDMM, self).__init__(**kwargs)
        initlocals = locals()
        initlocals.pop("self")
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

    def run(self):
        return
        # ret, x = FISTA_algorithm(self.i_guess, self.obj, self.fx, self.gx, self.gradfx, self.gprox,
        # self.eta, self.maxiter, self.tol, self.verbose)
        # converged, error =
        # return
