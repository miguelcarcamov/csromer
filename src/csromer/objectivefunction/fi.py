"""
Base class for objective function terms (Pyralysis-style interface with backward compatibility).
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from ..dictionaries import Wavelet

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(init=True, repr=True)
class Fi(metaclass=ABCMeta):
    """
    Base class for objective function terms. Supports both legacy API
    (evaluate, calculate_gradient, calculate_prox) and Pyralysis-style API
    (function, gradient, prox with rho).
    """
    reg: float = None
    norm_factor: float = None
    wavelet: Wavelet = None
    # Pyralysis-style attributes (backward compatible: reg is the storage)
    is_differentiable: bool = True
    persist_proximal: bool = False
    _func_value: float = field(init=False, repr=False, default=0.0)
    _grad_value: Union[np.ndarray, Any] = field(init=False, repr=False, default=None)
    parameter: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.reg is None:
            self.reg = 1.0
        if self.norm_factor is None:
            self.norm_factor = 1.0

    @property
    def penalization_factor(self) -> float:
        """Penalization factor (alias for reg for Pyralysis-style API)."""
        return self.reg

    @penalization_factor.setter
    def penalization_factor(self, value: float):
        self.reg = value

    @property
    def func_value(self) -> float:
        """Result of the last function evaluation."""
        return self._func_value

    @property
    def grad_value(self):
        """Result of the last gradient computation."""
        return self._grad_value

    # --- Legacy API (unchanged) ---
    @abstractmethod
    def evaluate(self, x):
        """Evaluate the term at x. Legacy API."""
        pass

    @abstractmethod
    def calculate_gradient(self, x):
        """Gradient at x. Legacy API."""
        pass

    @abstractmethod
    def calculate_prox(self, x, nu):
        """Proximal operator at x with step nu. Legacy API."""
        pass

    # --- Pyralysis-style API ---
    def function(self, *, mask=None):
        """
        Compute the function value. Uses parameter.data if parameter is set,
        otherwise the term must be used with evaluate(x) directly.
        """
        if self.parameter is not None and self.parameter.data is not None:
            x = self.parameter.data
        else:
            raise ValueError("Fi.function() requires parameter.data to be set")
        val = self.evaluate(x)
        self._func_value = float(val) if hasattr(val, "item") else float(val)
        return val

    def gradient(self, iter=1, *, mask=None):
        """
        Compute the gradient. Uses parameter.data if parameter is set.
        """
        if self.parameter is not None and self.parameter.data is not None:
            x = self.parameter.data
        else:
            raise ValueError("Fi.gradient() requires parameter.data to be set")
        g = self.calculate_gradient(x)
        self._grad_value = g
        return g

    def prox(self, x=None, rho: float = 1.0):
        """
        Proximal operator with penalty parameter rho.
        If x is None, uses parameter.data. Maps rho to nu for legacy calculate_prox.
        """
        if x is None and self.parameter is not None:
            x = self.parameter.data
        if x is None:
            raise ValueError("Fi.prox() requires x or parameter.data")
        rho_val = float(rho)
        if da is not None and hasattr(rho_val, "compute"):
            rho_val = float(rho_val.compute())
        result = self._prox_impl(x, rho_val)
        if self.persist_proximal and da is not None and isinstance(result, da.Array):
            result = result.persist()
        return result

    def _prox_impl(self, x, rho: float = 1.0):
        """
        Implementation of the proximal operator. Default delegates to calculate_prox(x, nu=rho).
        Subclasses can override to customize behavior.
        """
        return self.calculate_prox(x, nu=rho)
