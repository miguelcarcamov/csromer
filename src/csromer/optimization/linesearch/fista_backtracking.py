"""
FISTA backtracking line search: find L such that F(prox(y - (1/L)*grad)) <= Q_L (Pyralysis-style).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from ...utils.array_utils import maybe_compute
from .linesearcher import LineSearcher


@dataclass(init=True, repr=True)
class FISTABacktracking(LineSearcher):
    """
    FISTA backtracking: find L such that F(x_k) <= Q_L(x_k, y_k) where
    x_k = prox_{g/L}(y_k - (1/L)*grad(y_k)), Q_L = f(y) + <x-y, grad> + (L/2)*||x-y||^2 + g(x).
    """

    initial_lipschitz: float = 1.0
    decrease: float = 0.5
    _prev_lipschitz: float = field(default=None, init=False, repr=False)

    def _get_proximal(self):
        """Proximal from non-differentiable term (single term for FISTA)."""
        non_diff = [t for t in self.objective_function.F if not getattr(t, "is_differentiable", True)]
        if len(non_diff) == 0:
            return lambda x, rho: x
        if len(non_diff) == 1:
            def prox(x, rho):
                step = 1.0 / rho if rho != 0 else 1.0
                return non_diff[0].calculate_prox(x, nu=step)
            return prox
        raise ValueError("FISTA backtracking requires 0 or 1 non-differentiable terms")

    def _g_at(self, x_array) -> float:
        """Evaluate non-differentiable terms at x."""
        return self.objective_function.calculate_function(
            x_array, nondifferentiable_only=True
        )

    def _compute_Q_L(self, x_k, y_k, f_y, grad_y, L) -> float:
        """Q_L(x,y) = f(y) + <x-y, grad(y)> + (L/2)*||x-y||^2 + g(x)."""
        diff = x_k - y_k
        inner = np.real(np.vdot(np.ravel(diff), np.ravel(grad_y)))
        inner = float(maybe_compute(inner))
        norm_sq = np.real(np.vdot(np.ravel(diff), np.ravel(diff)))
        norm_sq = float(maybe_compute(norm_sq))
        q_smooth = f_y + inner + (L / 2.0) * norm_sq
        g_x = self._g_at(x_k)
        return q_smooth + g_x

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        """y_k = x.data; find L, return (F(x_k), 1/L) where x_k = prox(y - (1/L)*grad)."""
        self._read_kwargs(**kwargs)
        y_k = np.array(x.data, copy=True)
        grad_f_y = self.objective_function.dphi
        if grad_f_y is None:
            raise ValueError("objective_function.dphi must be set (call calculate_gradient)")
        grad_f_y = np.asarray(grad_f_y)
        f_y = self.objective_function.calculate_function(y_k, differentiable_only=True)
        F_y = self.objective_function.evaluate(y_k)
        F_y = float(maybe_compute(F_y)) if hasattr(F_y, "compute") else float(np.asarray(F_y).item())
        prox = self._get_proximal()
        # Warm start (Pyralysis-style): try larger step (smaller L) than last time
        # First iteration: use initial_lipschitz only (e.g. 1.0) so we try full gradient step; backtracking increases L until bound holds
        if self._prev_lipschitz is not None:
            lipschitz_L = max(self.initial_lipschitz, self._prev_lipschitz * 0.5)
        else:
            lipschitz_L = self.initial_lipschitz
        for _ in range(self.max_iter):
            gradient_step = y_k - (1.0 / lipschitz_L) * grad_f_y
            x_k_candidate = prox(gradient_step, lipschitz_L)
            x_k_candidate = np.asarray(x_k_candidate)
            full_F = self.objective_function.evaluate(x_k_candidate)
            full_F = float(maybe_compute(full_F)) if hasattr(full_F, "compute") else float(np.asarray(full_F).item())
            Q_L = self._compute_Q_L(x_k_candidate, y_k, f_y, grad_f_y, lipschitz_L)
            # Require F <= Q_L (bound) and F(x) <= F(y) (monotone step for MFISTA)
            if np.isfinite(full_F) and full_F <= Q_L and full_F <= F_y:
                break
            lipschitz_L /= self.decrease
            if lipschitz_L > 1e10:
                break
        self._prev_lipschitz = lipschitz_L
        x.data = x_k_candidate
        return full_F, 1.0 / lipschitz_L

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "initial_lipschitz" in kwargs:
            self.initial_lipschitz = kwargs["initial_lipschitz"]
        if "decrease" in kwargs:
            self.decrease = kwargs["decrease"]
