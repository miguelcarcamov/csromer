"""
Non-linear Conjugate Gradient optimizer (Pyralysis-style OOP design).
Base class ConjugateGradient with variant subclasses: FletcherReeves, PolakRibiere,
HestenesStiefel, DaiYuan, HagerZhang. Supports dask arrays and Powell restart.
"""
from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from ...reconstruction.parameter import Parameter
from ...utils.array_utils import math_module, maybe_compute
from ..optimizer import Optimizer
from .gradient_optimizer import GradientOptimizer, _inner, _norm2

try:
    import dask.array as da
except ImportError:
    da = None


# Powell (1977) restart: restart when g_{k+1}^T g_k < eta * ||g_{k+1}||^2
POWELL_RESTART_ETA = 0.2


class GradientNormError(Exception):
    """
    Raised when gradient norm is zero (degenerate step).

    Exception class for CG optimization errors.
    """
    pass


@dataclass(init=True, repr=True)
class ConjugateGradient(GradientOptimizer):
    """
    Non-linear Conjugate Gradient for smooth unconstrained minimization.

    Base class for CG variants. Search direction: d_{k+1} = -g_{k+1} + beta_k * d_k.
    Uses Powell restart and optional negative-beta restart. Supports dask arrays.
    """

    def run(self) -> Tuple[float, Parameter]:
        """
        Run CG optimization.

        Public method. Performs conjugate gradient iterations with line search
        and restart logic.

        Returns:
            Tuple of (final_cost, optimized_parameter)

        Raises:
            ValueError: If guess_param or F_obj is None
        """
        if self.guess_param is None or self.F_obj is None:
            raise ValueError("guess_param and F_obj cannot be None")

        current_param, prev_function_value, prev_gradient = self._initialize_optimization_state(
            self.guess_param
        )
        prev_search_direction = -np.asarray(prev_gradient, dtype=current_param.data.dtype)
        if da is not None and hasattr(prev_gradient, "chunks"):
            prev_search_direction = da.asarray(prev_search_direction)

        if self.verbose:
            print(f"Starting {self.method_name()} (Conjugate Gradient)")
            print(f"Initial function value = {prev_function_value:.6f}")

        max_iter = self.maxiter or 5000
        for iteration in range(max_iter):
            (
                current_param,
                new_function_value,
                current_gradient,
                new_search_direction,
                converged,
            ) = self._perform_iteration(
                iteration,
                current_param,
                prev_gradient,
                prev_search_direction,
            )

            if converged:
                return new_function_value, current_param

            if self._check_function_convergence(new_function_value, prev_function_value):
                if self.verbose:
                    print(f"{self.method_name()} converged after {iteration + 1} iterations")
                return new_function_value, current_param

            prev_function_value = new_function_value
            prev_gradient = current_gradient
            prev_search_direction = new_search_direction

        if self.verbose:
            print(f"{self.method_name()} reached max iterations ({max_iter})")
        return prev_function_value, current_param

    def _line_search(
        self, x, d, f_x: float, grad_x, c1: float = None, rho: float = None, max_ls: int = 30
    ) -> float:
        """
        Armijo backtracking line search.

        Protected method. Finds alpha such that f(x + alpha*d) <= f(x) + c1*alpha*<grad,d>.

        Args:
            x: Current point
            d: Search direction
            f_x: Current function value
            grad_x: Current gradient
            c1: Armijo parameter (default: self.c1)
            rho: Backtracking factor (default: self.rho)
            max_ls: Maximum line search iterations

        Returns:
            Step size alpha
        """
        c1 = c1 if c1 is not None else self.c1
        rho = rho if rho is not None else self.rho
        xp = math_module(x)
        slope = _inner(grad_x, d)
        if slope >= 0:
            return 0.0
        alpha = 1.0
        for _ in range(max_ls):
            x_new = x + alpha * d
            f_new = self.F_obj.evaluate(x_new)
            if hasattr(f_new, "compute"):
                f_new = float(f_new.compute())
            else:
                f_new = float(np.asarray(f_new).item())
            if f_new <= f_x + c1 * alpha * slope:
                return alpha
            alpha *= rho
        return alpha

    def conjugate_gradient_parameter(
        self, grad, grad_prev, dir_prev
    ) -> Tuple[float, float, float]:
        """
        Compute beta_k and scalars for restart check.

        Public method. Computes conjugate gradient parameter beta and auxiliary
        scalars for restart logic.

        Args:
            grad: Current gradient
            grad_prev: Previous gradient
            dir_prev: Previous search direction

        Returns:
            Tuple of (beta, g_dot_g_prev, norm2_g)

        Raises:
            GradientNormError: If ||grad_prev||^2 == 0
        """
        norm2_grad_prev = _norm2(grad_prev)
        if norm2_grad_prev == 0.0:
            raise GradientNormError("Previous gradient norm is zero")

        g_dot_g_prev = _inner(grad, grad_prev)
        norm2_grad = _norm2(grad)

        beta = self._conjugate_gradient_parameter(
            grad,
            grad_prev,
            dir_prev=dir_prev,
            norm2_grad_prev=norm2_grad_prev,
            norm2_grad=norm2_grad,
        )
        if hasattr(beta, "compute"):
            beta = float(maybe_compute(beta))
        else:
            beta = float(np.asarray(beta).item())
        return beta, g_dot_g_prev, norm2_grad

    def _should_restart(
        self, conjugate_parameter: float, g_dot_g_prev: float, norm2_g: float
    ) -> bool:
        """
        Check if restart is needed (Powell condition or negative beta).

        Protected method. Restarts with steepest descent when beta <= 0 or
        Powell condition holds.

        Args:
            conjugate_parameter: Beta value
            g_dot_g_prev: <g_k+1, g_k>
            norm2_g: ||g_k+1||^2

        Returns:
            True if restart needed
        """
        if conjugate_parameter <= 0.0:
            return True
        return norm2_g > 0 and g_dot_g_prev < POWELL_RESTART_ETA * norm2_g

    def _perform_iteration(
        self,
        iteration: int,
        current_param: Parameter,
        prev_gradient,
        prev_search_direction,
    ) -> Tuple[Parameter, float, any, Optional[any], bool]:
        """
        Perform one CG iteration.

        Protected method. Performs line search, updates parameter, computes gradient,
        beta, and new search direction.

        Args:
            iteration: Iteration number
            current_param: Current parameter
            prev_gradient: Previous gradient
            prev_search_direction: Previous search direction

        Returns:
            Tuple of (updated_param, new_f, current_gradient, new_search_direction, converged)
        """
        if self.verbose:
            print(f"Iteration {iteration + 1}")

        x = np.array(current_param.data, copy=False)
        f_x = self.F_obj.evaluate(x)
        if hasattr(f_x, "compute"):
            f_x = float(f_x.compute())
        else:
            f_x = float(np.asarray(f_x).item())
        grad_x = self._grad(x)

        alpha = self._line_search(x, prev_search_direction, f_x, grad_x)
        x_new = x + alpha * prev_search_direction

        updated_param = copy.deepcopy(current_param)
        updated_param.data = x_new

        new_function_value = self.F_obj.evaluate(x_new)
        if hasattr(new_function_value, "compute"):
            new_function_value = float(new_function_value.compute())
        else:
            new_function_value = float(np.asarray(new_function_value).item())

        current_gradient = self._grad(x_new)

        if self._check_gradient_convergence(
            updated_param, current_gradient, new_function_value, self.verbose
        ):
            return updated_param, new_function_value, current_gradient, None, True

        try:
            beta, g_dot_g_prev, norm2_g = self.conjugate_gradient_parameter(
                current_gradient, prev_gradient, prev_search_direction
            )
        except GradientNormError:
            if self.verbose:
                print("Exit due to zero gradient norm")
            return updated_param, new_function_value, current_gradient, None, True

        if self._should_restart(beta, g_dot_g_prev, norm2_g):
            beta = 0.0

        xp = math_module(current_gradient)
        new_search_direction = -np.asarray(current_gradient, dtype=x_new.dtype) + beta * np.asarray(
            prev_search_direction, dtype=x_new.dtype
        )
        if da is not None and hasattr(current_gradient, "chunks"):
            new_search_direction = xp.asarray(new_search_direction)

        return (
            updated_param,
            new_function_value,
            current_gradient,
            new_search_direction,
            False,
        )

    @abstractmethod
    def method_name(self) -> str:
        """
        Name of the CG variant.

        Abstract method: subclasses must implement.

        Returns:
            Method name (string)
        """
        raise NotImplementedError

    @abstractmethod
    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev: float = None, norm2_grad: float = None
    ) -> Union[float, any]:
        """
        Compute beta_k for this variant.

        Protected abstract method: subclasses must implement. May return scalar
        or array (will be computed to float).

        Args:
            grad: Current gradient
            grad_prev: Previous gradient
            dir_prev: Previous search direction
            norm2_grad_prev: ||grad_prev||^2 (precomputed)
            norm2_grad: ||grad||^2 (precomputed)

        Returns:
            Beta value (scalar or array)
        """
        raise NotImplementedError


# --- Variant subclasses (Pyralysis-style) ---


@dataclass(init=True, repr=True)
class FletcherReeves(ConjugateGradient):
    """
    Fletcher-Reeves CG variant.

    Beta formula: beta = ||g_{k+1}||^2 / ||g_k||^2.
    """

    def method_name(self) -> str:
        """
        Return method name.

        Returns:
            "Fletcher-Reeves"
        """
        return "Fletcher-Reeves"

    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev=None, norm2_grad=None
    ):
        """
        Compute Fletcher-Reeves beta.

        Protected method.
        """
        if norm2_grad is None:
            norm2_grad = _norm2(grad)
        return norm2_grad / max(norm2_grad_prev, 1e-20)


@dataclass(init=True, repr=True)
class PolakRibiere(ConjugateGradient):
    """
    Polak-Ribière-Polyak CG variant.

    Beta formula: beta = g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2.
    """

    def method_name(self) -> str:
        """
        Return method name.

        Returns:
            "Polak-Ribiere-Polyak"
        """
        return "Polak-Ribiere-Polyak"

    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev=None, norm2_grad=None
    ):
        """
        Compute Polak-Ribière beta.

        Protected method.
        """
        xp = math_module(grad)
        grad_diff = grad - grad_prev
        numer = _inner(grad, grad_diff)
        return numer / max(norm2_grad_prev, 1e-20)


@dataclass(init=True, repr=True)
class HestenesStiefel(ConjugateGradient):
    """
    Hestenes-Stiefel CG variant.

    Beta formula: beta = g_{k+1}^T (g_{k+1} - g_k) / (d_k^T (g_{k+1} - g_k)).
    """

    def method_name(self) -> str:
        """
        Return method name.

        Returns:
            "Hestenes-Stiefel"
        """
        return "Hestenes-Stiefel"

    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev=None, norm2_grad=None
    ):
        """
        Compute Hestenes-Stiefel beta.

        Protected method.
        """
        grad_diff = grad - grad_prev
        numer = _inner(grad, grad_diff)
        denom = _inner(dir_prev, grad_diff)
        if abs(denom) < 1e-20:
            return 0.0
        return numer / denom


@dataclass(init=True, repr=True)
class DaiYuan(ConjugateGradient):
    """
    Dai-Yuan CG variant.

    Beta formula: beta = ||g_{k+1}||^2 / (d_k^T (g_{k+1} - g_k)).
    """

    def method_name(self) -> str:
        """
        Return method name.

        Returns:
            "Dai-Yuan"
        """
        return "Dai-Yuan"

    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev=None, norm2_grad=None
    ):
        """
        Compute Dai-Yuan beta.

        Protected method.
        """
        if norm2_grad is None:
            norm2_grad = _norm2(grad)
        grad_diff = grad - grad_prev
        denom = _inner(dir_prev, grad_diff)
        if abs(denom) < 1e-20:
            return 0.0
        return norm2_grad / denom


@dataclass(init=True, repr=True)
class HagerZhang(ConjugateGradient):
    """
    Hager-Zhang CG variant.

    Beta formula: beta = (1/(d^T y)) * (y - 2*d*||y||^2/(d^T y))^T g_{k+1},
    where y = g_{k+1} - g_k.
    """

    def method_name(self) -> str:
        """
        Return method name.

        Returns:
            "Hager-Zhang"
        """
        return "Hager-Zhang"

    def _conjugate_gradient_parameter(
        self, grad, grad_prev, *, dir_prev=None, norm2_grad_prev=None, norm2_grad=None
    ):
        """
        Compute Hager-Zhang beta.

        Protected method.
        """
        grad_diff = grad - grad_prev
        denom = _inner(dir_prev, grad_diff)
        if abs(denom) < 1e-20:
            return 0.0
        norm2_y = _norm2(grad_diff)
        term = grad_diff - 2 * (norm2_y / denom) * dir_prev
        numer = _inner(term, grad)
        return numer / denom
