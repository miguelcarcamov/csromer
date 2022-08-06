#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import proxmin as pmin
from scipy.optimize import minimize

from ..reconstruction.parameter import Parameter
from .methods.fista import FISTA_algorithm
from .methods.sdmm import sdmm


class Optimizer(metaclass=ABCMeta):

    def __init__(
        self,
        guess_param: Parameter = None,
        F_obj=None,
        maxiter: int = None,
        method=None,
        tol: float = np.finfo(np.float32).tiny,
        verbose: bool = True,
    ):
        self.guess_param = guess_param
        self.F_obj = F_obj
        self.maxiter = maxiter
        self.method = method
        self.tol = tol
        self.verbose = verbose

    @abstractmethod
    def run(self):
        return


class FixedPointMethod(Optimizer):

    def __init__(self, gx=None, **kwargs):
        super().__init__(**kwargs)
        self.gx = gx

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
        super().__init__(**kwargs)
        self.method = method

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
        super().__init__(**kwargs)
        self.fx = fx
        self.gx = gx
        self.noise = noise

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
        super().__init__(**kwargs)
        self.fx = fx
        self.gx = gx
        self.L0 = L0

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
        super().__init__(**kwargs)
        self.fx = fx
        self.gx = gx
        self.gradfx = gradfx
        self.gprox = gprox
        self.eta = eta

    def run(self):
        return
        # ret, x = FISTA_algorithm(self.i_guess, self.obj, self.fx, self.gx, self.gradfx, self.gprox,
        # self.eta, self.maxiter, self.tol, self.verbose)
        # converged, error =
        # return
