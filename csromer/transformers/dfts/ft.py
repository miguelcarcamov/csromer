#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""
from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import speed_of_light as c

from ...reconstruction import Parameter

if TYPE_CHECKING:
    from ..base import Dataset


class FT(metaclass=ABCMeta):

    def __init__(self, dataset: Dataset = None, parameter: Parameter = None, use_weights=True):
        self.dataset = dataset
        self.s = None
        self.use_weights = use_weights

        if self.dataset is not None:
            if self.use_weights:
                self.weights = self.dataset.w
            else:
                self.weights = np.ones_like(self.dataset.w)
            self.k = np.sum(self.weights)
        else:
            self.weights = None
            self.k = None

        if parameter is not None:
            self.parameter = copy.deepcopy(parameter)
        else:
            self.parameter = None

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def forward_normalized(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def RMTF(self):
        pass
