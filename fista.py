#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:09:14 2019

@author: miguel
"""
import numpy as np
from prox_functions import prox_function
def calc_Q(x, y, L, fx, gx, grad):
    res = fx(y) + np.dot(x-y, grad(y)) + (L/2)*(np.linalg.norm(x-y)**2) + gx(x)
    return res

def FISTA(x_init, F, fx, gx, grad, prox, eta, max_iter, tol, verbose):
    
    x_old = x_init
    y_old = x_init
    t_old = 1
    L = 1
    temp_prox = prox_function(prox.reg)
    
    for iter in range(0, max_iter):
        Lbar = L;
        while True:
            temp_prox.set_reg(prox.reg/Lbar)
            zk = prox.soft_thresholding(y_old - (1/Lbar)*grad(y_old))
            F_ret = F(zk)
            Q = calc_Q(zk, y_old, Lbar, fx, gx, grad)

            if F_ret <= Q:
                break
            Lbar = Lbar*eta
            L = Lbar
            
        prox.set_reg(prox.reg/L)
        x_new = prox.soft_thresholding(y_old- (1/L)*grad(y_old))
        t_new = 0.5*(1+ np.sqrt(1 + 4*t_old**2))
        dif = x_new - x_old
        y_new = x_new + (t_old - 1)/t_new * (dif)
        #check stop criteria
        e = np.linalg.norm(dif,ord=1)/len(x_new)
        
        if e < tol:
            print("Exit due to tolerance")
            break
        
        # If stop criteria is not met then update variables
        x_old = x_new
        t_old = t_new
        y_old = y_new
        
        if verbose:
            cost = F(x_new)
            print("Iteration: ", iter, " objective function value: ", cost)
            print("L value: ", L)

    min_cost = F(x_new)
    return min_cost, x_new         
    
    
    