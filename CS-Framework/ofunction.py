#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:43:15 2019

@author: miguel
"""

import numpy as np


class OFunction:
    def __init__(self, F=[]):
        initlocals = locals()
        initlocals.pop('self')
        for a_attribute in initlocals.keys():
            setattr(self, a_attribute, initlocals[a_attribute])

        self.values = np.zeros(len(F))
        if len(F) == 0:
            self.prox_functions = []
        else:
            self.prox_functions = [f_i for f_i in self.F]

    def getProxFunctions(self):
        return self.prox_functions

    def getValues(self):
        return self.values

    def getLambda(self, id=0):
        return self.F[id].getRegularizationParameter()

    def setLambda(self, reg=0.0, id=0):
        self.F[id].setRegularizationParameter(reg=reg)

    def evaluate(self, x):
        ret = 0.0

        for i in range(0, len(self.F)):
            self.values[i] = self.F[i].evaluate(x)
            ret += self.F[i].reg * self.values[i]
        return ret

    def calculate_gradient(self, x):
        res = np.zeros(len(x), dtype=x.dtype)
        for f_i in self.F:
            res += f_i.reg * f_i.calculate_gradient(x)
        return res

    def calc_prox(self, x, nu=0, id=0):
        f_i = self.F[id]
        return f_i.calculate_prox(x, nu)
