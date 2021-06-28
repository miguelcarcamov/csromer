#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:09:14 2019

@author: miguel
"""
import numpy as np
import sys


def FISTA_algorithm(x=None, F=None, fx=None, g_prox=None, max_iter=None, tol=1e-12, n=None, noise=None,
                    verbose=True):
    if x is None and n is not None:
        x = np.zeros(n) + 1j * np.zeros(n)
    t = 1
    z = x.copy()
    min_cost = 0.0

    if max_iter is None and noise is not None:
        max_iter = int(np.floor(g_prox.getLambda() / noise))
        if verbose:
            print("Iterations set to " + str(max_iter))

    if noise is None:
        noise = 0.0

    if noise >= g_prox.getLambda():
        print("Error, noise cannot be greater than lambda")
        return min_cost, x

    for it in range(0, max_iter):
        xold = x.copy()
        z = z - fx(z)
        x = g_prox.calc_prox(z)

        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        e = np.sum(np.abs(x - xold)) / len(x)
        if e <= tol:
            print("Exit due to tolerance: ", e, " < ", tol)
            print("Iterations: ", it)
            break

        if verbose and it % 50 == 0:
            cost = F(x)
            print("Iteration: ", it,
                  " objective function value: {0:0.5f}".format(cost))
        new_lambda = g_prox.getLambda() - noise
        if new_lambda > 0.0:
            g_prox.setLambda(reg=new_lambda)
        else:
            print("Exit due to negative regularization parameter")
            break
    min_cost = F(x)
    return min_cost, x
