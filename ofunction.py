#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:43:15 2019

@author: miguel
"""

import numpy as np
from utilities import real_to_complex, complex_to_real

class OFunction:
    value = 0.0
    priors = []
    def __init__(self, b, dft_obj, priors=[]):
        self.b = b
        self.dft = dft_obj
        self.priors = priors
        
    def evaluate(self, x):
        x_complex = real_to_complex(x)

        y = self.b - self.dft.forward(x_complex)

        ret = 0.5 * (np.linalg.norm(y)**2)
        
        for p in self.priors:
            ret += p.reg * p.evaluate(x_complex)
            
        return ret
    
    def calculate_gradient(self, x):
        
        res = complex_to_real(self.dft.backward_dirty(self.b)) - x
        
        for p in self.priors:
            res += p.reg * p.calculate_gradient(x)
        
        return res
        