"""
Base class for gradient-based optimizers (Pyralysis-style).
Provides gradient/function convergence checks and initialization.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from ...utils.array_utils import math_module
from ..optimizer import Optimizer


def _inner(a, b) -> float:
    """
    Inner product; works with dask by computing to scalar.

    Private helper function. Computes <a, b> and returns real part.

    Args:
        a: First array
        b: Second array

    Returns:
        Inner product (float, real part)
    """
    out = np.vdot(np.ravel(a), np.ravel(b))
    return float(np.real(out.compute())) if hasattr(out, "compute") else float(np.real(out))


def _norm2(a) -> float:
    """
    Squared L2 norm; works with dask.

    Private helper function. Computes ||a||^2.

    Args:
        a: Input array

    Returns:
        Squared L2 norm (float)
    """
    out = np.vdot(np.ravel(a), np.ravel(a))
    return float(np.real(out.compute())) if hasattr(out, "compute") else float(np.real(out))


@dataclass(init=True, repr=True)
class GradientOptimizer(Optimizer):
    """
    Base class for gradient-based optimizers.

    Provides gradient and function convergence checks, line search parameters,
    and initialization helpers. Uses gradient and function tolerances for convergence.

    Attributes:
        grad_fun: Optional gradient function (default: F_obj.calculate_gradient)
        gtol: Gradient tolerance (default: 1e-8)
        c1: Armijo parameter for line search (default: 1e-4)
        rho: Backtracking factor for line search (default: 0.5)
    """

    grad_fun: Optional[Callable] = None
    gtol: float = 1e-8
    c1: float = 1e-4
    rho: float = 0.5

    def _grad(self, x):
        """
        Compute gradient at x.

        Protected method. Uses grad_fun if provided, otherwise F_obj.calculate_gradient.

        Args:
            x: Input array

        Returns:
            Gradient array
        """
        if self.grad_fun is not None:
            return self.grad_fun(x)
        return self.F_obj.calculate_gradient(x)

    def _condition(self, parameter: Parameter, gradient, function_value: float) -> float:
        """
        Compute scaled gradient stopping condition (Pyralysis-style).

        Protected method. Computes max(|grad| * max(|param|, 1)) / max(|f|, 1).

        Args:
            parameter: Parameter object
            gradient: Gradient array
            function_value: Current function value

        Returns:
            Scaled gradient condition (float)
        """
        xp = math_module(gradient)
        abs_param = xp.abs(parameter.data)
        div = max(float(function_value), 1.0)
        condition = xp.abs(gradient) * xp.maximum(abs_param, 1.0) / div
        max_val = xp.max(condition)
        return float(max_val.compute()) if hasattr(max_val, "compute") else float(max_val)

    def _check_function_convergence(self, func_current: float, func_previous: float) -> bool:
        """
        Check relative function change convergence.

        Protected method. Returns True if relative function change <= tol.

        Args:
            func_current: Current function value
            func_previous: Previous function value

        Returns:
            True if converged
        """
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
        """
        Check scaled gradient norm convergence.

        Protected method. Returns True if scaled gradient norm < gtol.

        Args:
            parameter: Parameter object
            gradient: Gradient array
            function_value: Current function value
            verbose: Verbose output flag
            gtol: Gradient tolerance (default: self.gtol)

        Returns:
            True if converged
        """
        gcond = self._condition(parameter, gradient, function_value)
        g_tol = gtol if gtol is not None else self.gtol
        if gcond < g_tol:
            if verbose:
                print("Exit due to gradient tolerance")
            return True
        return False

    def _initialize_optimization_state(self, parameter: Parameter) -> Tuple[Parameter, float, any]:
        """
        Compute initial function value and gradient.

        Protected method. Evaluates objective and gradient at initial point.

        Args:
            parameter: Initial parameter

        Returns:
            Tuple of (param, f_value, gradient)
        """
        x = np.array(parameter.data, copy=False)
        f_value = self.F_obj.evaluate(x)
        if hasattr(f_value, "compute"):
            f_value = float(f_value.compute())
        else:
            f_value = float(np.asarray(f_value).item())
        gradient = self._grad(x)
        return parameter, f_value, gradient
