#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:09:14 2019

@author: miguel
"""
import numpy as np


def calc_Q(x, y, L, fx, gx, grad):
    res = fx(y) + np.dot(x - y, grad(y)) + gx(x) + \
        (1 / (2 * L)) * (np.linalg.norm(x - y)**2)
    return res


def FISTA(x_init, F, fx, gx, grad, g_prox, eta, max_iter, tol, verbose):

    x_old = x_init
    y_old = x_init
    L = 1

    for iter in range(0, max_iter):
        Lbar = L
        while True:
            # temp_prox.set_reg(prox.reg/Lbar)
            y_eval = y_old - Lbar * grad(y_old)
            zk = g_prox.calc_prox(g_prox.calc_prox(
                y_old, y_eval, 0), id=1)

            F_ret = fx(zk)
            Q = calc_Q(zk, y_old, Lbar, fx, gx, grad)

            if F_ret <= Q:
                break
            else:
                Lbar = Lbar * eta
                L = Lbar
        # print("L value: ", L)
        # prox.set_reg(prox.reg/L)
        x_new = zk
        # t_new = 0.5*(1+ np.sqrt(1 + 4*t_old**2))

        t_min = min(1.0, eta / (1 / L))
        dif = x_new - x_old
        e = 2 * (np.linalg.norm(dif)**2) / (t_min * (iter + 1)**2)
        # check stop criteria
        if e <= tol:
            print("Exit due to tolerance: ", e, " < ", tol)
            break

        # If stop criteria is not met then update variables
        k_iter = iter / (iter + 3)
        y_new = x_new + k_iter * dif
        x_old = x_new
        y_old = y_new

        if verbose and iter % 50 == 0:
            cost = F(x_new)
            print("Iteration: ", iter,
                  " objective function value: {0:0.5f}".format(cost))
    min_cost = F(x_new)
    return min_cost, x_new
