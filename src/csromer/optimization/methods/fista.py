"""
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
Uses only F_obj (Pyralysis-style): gradient from differentiable terms, prox from
non-differentiable terms. Supports monotone FISTA (MFISTA) and adaptive restart.
Step size: use FISTABacktracking (default) for adaptive L, or fixed step when step= is set.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np

from ..optimizer import Optimizer
from ..linesearch import FISTABacktracking

if TYPE_CHECKING:
    from csromer.reconstruction import Parameter


def _f_value(F, x) -> float:
    """
    Evaluate F(x) and return scalar (dask-safe).
    
    Private helper function. Handles both numpy and dask arrays.
    
    Args:
        F: Objective function callable
        x: Input array
        
    Returns:
        Function value (float)
    """
    v = F(x)
    return float(v.compute()) if hasattr(v, "compute") else float(np.asarray(v).item())


def _inner_real(a, b) -> float:
    """
    Real part of inner product (dask-safe).
    
    Private helper function. Computes real part of <a, b>.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Real part of inner product (float)
    """
    out = np.real(np.vdot(np.ravel(a), np.ravel(b)))
    return float(out.compute()) if hasattr(out, "compute") else float(np.real(out))


def _check_function_convergence(f_current: float, f_previous: float, tol: float) -> bool:
    """
    True if relative function change is <= tol (same criterion as GradientOptimizer).
    """
    eps = np.finfo(np.float64).tiny
    denom = abs(f_current) + abs(f_previous) + eps
    return 2.0 * abs(f_current - f_previous) <= tol * denom


@dataclass(init=True, repr=True)
class FISTA(Optimizer):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    
    Optimizes objectives F(x) = f(x) + g(x) with smooth f and proximal for g.
    Step size: by default uses FISTABacktracking to adaptively find L each iteration.
    Set step= to use a fixed step size instead.
    
    Attributes:
        noise: Noise level for cooling schedule (optional)
        monotonic: If True, use monotone FISTA (reject non-monotone steps)
        adaptive_restart: Restart strategy: "function", "gradient", or None
        step: If set, fixed gradient step (no backtracking). If None, use linesearcher.
        linesearcher: FISTABacktracking instance. If None and step is None, one is created.
    """

    noise: float = None
    monotonic: bool = False
    adaptive_restart: Optional[Literal["function", "gradient"]] = None
    step: float = None  # If set, use fixed step (no backtracking). If None, use linesearcher.
    linesearcher: Optional[FISTABacktracking] = None  # If None and step is None, create default FISTABacktracking.

    def run(self) -> Tuple[float, "Parameter"]:
        """
        Run FISTA optimization.
        
        Public method. Performs FISTA iterations with optional cooling schedule
        and adaptive restart. Step size: FISTABacktracking (default) or fixed step if step= is set.
        
        Returns:
            Tuple of (final_cost, optimized_parameter)
        """
        def grad_f(z):
            return self.F_obj.calculate_gradient(z, differentiable_only=True)

        max_iter = self.maxiter if self.maxiter is not None else 500

        use_backtracking = self.step is None
        if use_backtracking:
            ls = self.linesearcher
            if ls is None:
                # Start with L=1 so 1/L=1 (try full gradient step); backtracking increases L until F <= Q_L
                ls = FISTABacktracking(initial_lipschitz=1.0)
            ls.objective_function = self.F_obj
            param_ls = copy.deepcopy(self.guess_param)

            def step_callback(y):
                self.F_obj.calculate_gradient(y, differentiable_only=True)
                param_ls.data = y
                f_new, _ = ls.search(param_ls)
                return np.array(param_ls.data, copy=True), f_new
            if self.verbose:
                print("FISTA step: backtracking (FISTABacktracking)")
        else:
            step = self.step
            def step_callback(y):
                g = grad_f(y)
                z_step = y - step * g
                x = self.F_obj.apply_prox_nondiff(z_step, nu=step)
                return x, _f_value(self.F_obj.evaluate, x)
            if self.verbose:
                print("FISTA step size (fixed): {:.2e}".format(step))

        ret, x = self._fista_algorithm(
            self.guess_param.data,
            self.F_obj.evaluate,
            max_iter,
            self.tol,
            self.guess_param.n,
            self.verbose,
            self.monotonic,
            self.adaptive_restart,
            step_callback,
        )
        param = copy.deepcopy(self.guess_param)
        param.data = x
        return ret, param

    @staticmethod
    def _fista_algorithm(
        x=None,
        F=None,
        max_iter: int = 500,
        tol: float = np.finfo(np.float32).tiny,
        n: int = None,
        verbose: bool = True,
        monotonic: bool = False,
        adaptive_restart: Optional[Literal["function", "gradient"]] = None,
        step_callback=None,
    ) -> Tuple[float, np.ndarray]:
        """
        Core FISTA algorithm implementation (Pyralysis-style: no lambda cooling).
        
        step_callback(y) must return (x, f_new) where x = prox(y - step*grad) and f_new = F(x).
        """
        if x is None and n is not None:
            x = np.zeros(n, dtype=np.complex64)
        x = np.array(x, copy=True)
        t = 1.0
        z = np.array(x, copy=True)

        f_prev = _f_value(F, x)
        if verbose:
            print("Initial function value = {:.6f}".format(f_prev))
        for it in range(0, max_iter):
            x_old = np.array(x, copy=True)
            y = np.array(z, copy=True)

            x, f_new = step_callback(y)

            progress = True

            if monotonic and f_new > f_prev:
                x = x_old
                z = x_old
                t = 1.0
                f_new = f_prev
                progress = False
                if verbose and it % 10 == 0:
                    print("Iteration: {} (monotone reject) objective: {:.5f}".format(it + 1, f_new))
            else:
                if adaptive_restart == "gradient":
                    inner = _inner_real(y - x, x - x_old)
                    if inner > 0:
                        t = 1.0
                        z = np.array(x, copy=True)
                elif adaptive_restart == "function" and not monotonic and f_new > f_prev:
                    t = 1.0
                    z = np.array(x, copy=True)

                t0 = t
                t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
                z = x + ((t0 - 1.0) / t) * (x - x_old)

            if _check_function_convergence(f_new, f_prev, tol):
                if verbose:
                    print("FISTA converged (relative function change <= tol) after {} iterations".format(it + 1))
                f_prev = f_new
                break

            f_prev = f_new

            if verbose and (it + 1) % 10 == 0 and progress:
                print("Iteration: {}  objective: {:.5f}".format(it + 1, f_new))

        return _f_value(F, x), x
