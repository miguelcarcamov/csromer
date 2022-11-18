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
from typing import TYPE_CHECKING, Union

import numpy as np

from ...reconstruction import Parameter

if TYPE_CHECKING:
    from ...base import Dataset


@dataclass(init=True, repr=True)
class FT(metaclass=ABCMeta):
    dataset: Dataset = None
    parameter: Parameter = None

    def __post_init__(self):

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
