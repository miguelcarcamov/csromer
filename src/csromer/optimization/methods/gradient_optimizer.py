"""
Base class for gradient-based optimizers (Pyralysis-style).
Provides gradient/function convergence checks and initialization.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ...utils.array_utils import math_module, maybe_compute
from ...reconstruction.parameter import Parameter
from ..optimizer import Optimizer


def _inner(a, b) -> float:
    """Inner product; works with dask by computing to scalar."""
    out = np.vdot(np.ravel(a), np.ravel(b))
    return float(np.real(maybe_compute(out)))


def _norm2(a) -> float:
    """Squared L2 norm; works with dask."""
    out = np.vdot(np.ravel(a), np.ravel(a))
    return float(np.real(maybe_compute(out)))


@dataclass(init=True, repr=True)
class GradientOptimizer(Optimizer):
    """
    Base class for gradient-based optimizers.
    Uses gradient and function tolerances for convergence.
    """

    grad_fun: Optional[Callable] = None
    gtol: float = 1e-8
    c1: float = 1e-4
    rho: float = 0.5

    def _grad(self, x):
        if self.grad_fun is not None:
            return self.grad_fun(x)
        return self.F_obj.calculate_gradient(x)

    def _condition(
        self, parameter: Parameter, gradient, function_value: float
    ) -> float:
        """Scaled gradient stopping condition (Pyralysis-style)."""
        xp = math_module(gradient)
        abs_param = xp.abs(parameter.data)
        div = max(float(function_value), 1.0)
        condition = xp.abs(gradient) * xp.maximum(abs_param, 1.0) / div
        max_val = xp.max(condition)
        return float(maybe_compute(max_val))

    def _check_function_convergence(
        self, func_current: float, func_previous: float
    ) -> bool:
        """Relative function change <= ftol (with epsilon)."""
        eps = np.finfo(np.float64).tiny
        denom = abs(func_current) + abs(func_previous) + eps
        return 2.0 * abs(func_current - func_previous) <= self.tol * denom

    def _check_gradient_convergence(
        self,
        parameter: Parameter,
        gradient,
        function_value: float,
        verbose: bool,
        gtol: float = None,
    ) -> bool:
        """True if scaled gradient norm < gtol."""
        gcond = self._condition(parameter, gradient, function_value)
        g_tol = gtol if gtol is not None else self.gtol
        if gcond < g_tol:
            if verbose:
                print("Exit due to gradient tolerance")
            return True
        return False

    def _initialize_optimization_state(
        self, parameter: Parameter
    ) -> Tuple[Parameter, float, any]:
        """Compute initial function value and gradient. Returns (param, f_value, gradient)."""
        x = np.array(parameter.data, copy=False)
        f_value = self.F_obj.evaluate(x)
        if hasattr(f_value, "compute"):
            f_value = float(f_value.compute())
        else:
            f_value = float(np.asarray(f_value).item())
        gradient = self._grad(x)
        return parameter, f_value, gradient
