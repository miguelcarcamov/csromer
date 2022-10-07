#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""
from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...reconstruction import Parameter

if TYPE_CHECKING:
    from ...base import Dataset


@dataclass(init=True, repr=True)
class FT(metaclass=ABCMeta):
    dataset: Dataset = None
    parameter: Parameter = None
    use_weights: bool = None
    k: float = field(init=False, default=1.0)
    weights: np.ndarray = field(init=False, default=None)

    def __post_init__(self):

        if self.use_weights is None:
            self.use_weights = True

        if self.dataset is not None:
            if self.use_weights:
                self.weights = self.dataset.w
            else:
                self.weights = np.ones_like(self.dataset.w)
            self.k = np.sum(self.weights)
        else:
            self.weights = None
            self.k = None

        if self.parameter is not None:
            self.parameter = copy.deepcopy(self.parameter)

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
