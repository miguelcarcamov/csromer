#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:09:14 2019

@author: miguel
"""
import numpy as np


def calc_Q(x, y, L, fx, gx, grad):
    res = fx(y) + np.dot(x - y, grad(y)) + gx(x) + L/2 * (np.sqrt(np.sum((x - y)**2)))
    return res


def FISTA_algorithm(x_init, F, fx, gx, grad, g_prox, eta, max_iter, tol, L0, verbose):

    x_old = x_init
    y_old = x_init
    t_old = 1
    L = L0

    for iter in range(0, max_iter):
        Lbar = L
        while True:
            # temp_prox.set_reg(prox.reg/Lbar)
            y_eval = y_old - 1/Lbar * grad(y_old)
            zk = g_prox.calc_prox(y_old, y_eval, 0)

            F_ret = fx(zk)
            Q = calc_Q(zk, y_old, Lbar, fx, gx, grad)

            if F_ret <= Q:
                break
            else:
                Lbar = Lbar * eta
                L = Lbar

        g_prox.setLambda(reg=g_prox.getLambda()/L)
        y_eval = y_old - 1/Lbar * grad(y_old)
        x_new = g_prox.calc_prox(y_old, y_eval, 0)

        t_new = 0.5*(1 + np.sqrt(1 + 4*t_old**2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);

        e = np.sum(np.abs(x_new-x_old))/len(x_new)
        if e <= tol:
            print("Exit due to tolerance: ", e, " < ", tol)
            print("Iterations: ", iter)
            break

        x_old = x_new;
        t_old = t_new;
        y_old = y_new;

        if verbose and iter % 50 == 0:
            cost = F(x_new)
            print("Iteration: ", iter,
                  " objective function value: {0:0.5f}".format(cost))
    min_cost = F(x_new)
    return min_cost, x_new
