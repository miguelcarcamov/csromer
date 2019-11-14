#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:46:13 2019

@author: miguel
"""
import numpy as np

class prox_function:
    def __init__(self, reg=None):
        self.reg = reg
    
    def set_reg(self, reg):
        self.reg = reg
        
    def soft_thresholding(self, x):
        j = np.abs(x) <= self.reg
        x[j] = 0
        j = np.abs(x) > self.reg
        x[j] = x[j] - np.sign(x[j])*self.reg
        return x
    
    def hard_thresholding(self, x):
        j = np.abs(x) < self.reg
        x[j] = 0
        return x