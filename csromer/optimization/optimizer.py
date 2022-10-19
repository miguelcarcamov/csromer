#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:51 2019

@author: miguel
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..objectivefunction import OFunction
from ..reconstruction.parameter import Parameter


@dataclass(init=True, repr=True)
class Optimizer(metaclass=ABCMeta):
    guess_param: Parameter = None
    F_obj: OFunction = None
    maxiter: int = None
    tol: float = field(init=True, default=np.finfo(np.float32).tiny)
    verbose: bool = None

    @abstractmethod
    def run(self):
        return
