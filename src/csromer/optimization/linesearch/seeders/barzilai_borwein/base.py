"""Barzilai-Borwein base seeder (Pyralysis-style). BB1 = ||s||^2/(s'y), BB2 = s'y/||y||^2."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..base import StepSizeSeeder


def _vdot_real(a, b) -> float:
    out = np.real(np.vdot(np.ravel(a), np.ravel(b)))
    return float(out.compute()) if hasattr(out, "compute") else float(np.real(out))


@dataclass(init=True, repr=True)
class BarzilaiBorwein(StepSizeSeeder):
    """Barzilai-Borwein step size using gradient history. s = x_k - x_{k-1}, y = g_k - g_{k-1}."""

    min_step: float = 1e-10
    init_step: float = 1.0
    curvature_tol: float = 1e-12

    _prev_x: Optional[object] = field(default=None, init=False, repr=False)
    _prev_g: Optional[object] = field(default=None, init=False, repr=False)
    _prev_step: Optional[float] = field(default=None, init=False, repr=False)
    _iter_count: int = field(default=0, init=False, repr=False)

    @staticmethod
    def _bb1(
        step_norm_squared: float, step_curvature: float, curvature_tol: float = 1e-12
    ) -> float:
        """Long BB: alpha = ||s||^2 / (s'y)."""
        if np.isfinite(step_curvature) and step_curvature > curvature_tol:
            return step_norm_squared / step_curvature
        return np.nan

    @staticmethod
    def _bb2(
        step_curvature: float,
        grad_change_norm_sqr: float,
        curvature_tol: float = 1e-12,
    ) -> float:
        """Short BB: alpha = s'y / ||y||^2."""
        if (
            np.isfinite(grad_change_norm_sqr) and grad_change_norm_sqr > curvature_tol
            and np.isfinite(step_curvature) and step_curvature > curvature_tol
        ):
            return step_curvature / grad_change_norm_sqr
        return np.nan

    @abstractmethod
    def _select_step_candidate(self, alpha_bb1: float, alpha_bb2: float,
                               iteration: int) -> Optional[float]:
        """Select between BB1 and BB2. Subclasses implement strategy."""
        raise NotImplementedError

    def estimate_step_size(self, x, objective_function) -> float:
        """Estimate step size using BB formulas and gradient history."""
        current_grad = objective_function.dphi
        current_step = x.data
        if current_grad is None:
            step_seed = self._prev_step if self._prev_step is not None else self.init_step
        elif self._prev_x is None or self._prev_g is None:
            step_seed = self._prev_step if self._prev_step is not None else self.init_step
        else:
            step_diff = np.asarray(current_step) - np.asarray(self._prev_x)
            grad_diff = np.asarray(current_grad) - np.asarray(self._prev_g)
            step_norm_sqr = _vdot_real(step_diff, step_diff)
            step_curvature = _vdot_real(step_diff, grad_diff)
            grad_change_norm_sqr = _vdot_real(grad_diff, grad_diff)
            alpha_bb1 = self._bb1(step_norm_sqr, step_curvature, curvature_tol=self.curvature_tol)
            alpha_bb2 = self._bb2(
                step_curvature, grad_change_norm_sqr, curvature_tol=self.curvature_tol
            )
            iteration = max(1, self._iter_count)
            alpha_candidate = self._select_step_candidate(alpha_bb1, alpha_bb2, iteration=iteration)
            if (
                alpha_candidate is not None and np.isfinite(alpha_candidate) and alpha_candidate > 0
            ):
                step_seed = alpha_candidate
            else:
                step_seed = self._prev_step if self._prev_step is not None else self.init_step
        step_seed = max(step_seed, self.min_step)
        self._prev_x = current_step
        self._prev_g = current_grad
        self._prev_step = step_seed
        self._iter_count += 1
        return step_seed
