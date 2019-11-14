#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:43:15 2019

@author: miguel
"""

import numpy as np

class OFunction:
    value = 0.0

    def __init__(self, F=[]):
        self.F = F
        
    def evaluate(self, x):
        ret = 0.0
        for f_i in self.F:
            ret += f_i.reg * f_i.evaluate(x)
            
        return ret
    
    def calculate_gradient(self, x):
        res = np.zeros(len(x), dtype=x.dtype)
        for f_i in self.F:
            res += f_i.reg * f_i.calculate_gradient(x)
        
        return res
        