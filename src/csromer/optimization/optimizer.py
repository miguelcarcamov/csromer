#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base optimizer class for iterative optimization algorithms.

Abstract interface that all optimizers must implement.
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..objectivefunction import OFunction
from ..reconstruction.parameter import Parameter


@dataclass(init=True, repr=True)
class Optimizer(metaclass=ABCMeta):
    """
    Base class for optimization algorithms.

    Abstract interface that defines the contract for optimizers. Subclasses must
    implement run() to perform the optimization.

    Attributes:
        guess_param: Initial parameter guess
        F_obj: Objective function (OFunction instance)
        maxiter: Maximum iterations (optional)
        tol: Tolerance for convergence (default: float32 tiny)
        verbose: Verbose output (optional)
    """
    guess_param: Parameter = None
    F_obj: OFunction = None
    maxiter: int = None
    tol: float = field(init=True, default=np.finfo(np.float32).tiny)
    verbose: bool = None

    @abstractmethod
    def run(self):
        """
        Run optimization.

        Abstract method: subclasses must implement. Performs optimization and
        returns optimized parameter.

        Returns:
            Tuple of (final_cost, optimized_parameter)
        """
        return
