#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:09:14 2019

@author: miguel
"""
import copy
from dataclasses import dataclass

import numpy as np

from ...objectivefunction import Chi2, Fi
from ..optimizer import Optimizer


@dataclass(init=True, repr=True)
class FISTA(Optimizer):

    fx: Chi2 = None
    gx: Fi = None
    noise: float = None

    def run(self):
        ret, x = self.__fista_algorithm(
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

    @staticmethod
    def __fista_algorithm(
        x=None,
        F=None,
        fx=None,
        g_prox=None,
        max_iter=None,
        tol=np.finfo(np.float32).tiny,
        n=None,
        noise=None,
        verbose=True,
    ):
        if x is None and n is not None:
            x = np.zeros(n, dtype=np.complex64)
        t = 1
        z = x.copy()
        min_cost = 0.0

        if max_iter is None and noise is not None:
            if noise is not np.nan:
                if noise != 0.0:
                    max_iter = int(np.floor(g_prox.getLambda() / noise))
                else:
                    noise = 1e-5
                    max_iter = int(np.floor(g_prox.getLambda() / noise))
            else:
                raise ValueError("Noise must be a number")
            if verbose:
                print("Iterations set to " + str(max_iter))

        if noise is None:
            noise = 1e-5

        if noise >= g_prox.getLambda():
            if verbose:
                print("Error, noise cannot be greater than lambda")
            return min_cost, x

        for it in range(0, max_iter):
            xold = x.copy()
            z = z - fx(z)
            x = g_prox.calc_prox(z)

            t0 = t
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
            z = x + ((t0 - 1.0) / t) * (x - xold)
            # e = np.sqrt(np.sum((x-xold)**2)) / np.sqrt(np.sum(xold**2))
            # print(e)
            e = np.sum(np.abs(x - xold)) / len(x)

            # if e <= tol:
            #    if verbose:
            #       print("Exit due to tolerance: ", e, " < ", tol)
            #    print("Iterations: ", it + 1)
            #    break

            if verbose and it % 10 == 0:
                cost = F(x)
                print("Iteration: ", it, " objective function value: {0:0.5f}".format(cost))
            new_lambda = g_prox.getLambda() - noise
            if new_lambda > 0.0:
                g_prox.setLambda(reg=new_lambda)
            else:
                if verbose:
                    print("Exit due to negative regularization parameter")
                break
        min_cost = F(x)
        return min_cost, x
