"""Quadratic interpolation step seeder (Pyralysis-style). Uses (0, f, g) and (alpha_prev, f_prev)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import StepSizeSeeder


def _vdot_real(a, b) -> float:
    out = np.real(np.vdot(np.ravel(a), np.ravel(b)))
    return float(out.compute()) if hasattr(out, "compute") else float(np.real(out))


@dataclass(init=True, repr=True)
class QuadraticInterpolationSeeder(StepSizeSeeder):
    """
    Quadratic interpolation along search direction.
    Fits (0, f_current, g_current) and (alpha_prev, f_prev); caller must call
    set_previous_step(alpha_used) after each line search.
    """

    max_step: float = 10.0
    _prev_f: Optional[float] = field(default=None, init=False, repr=False)
    _prev_step: Optional[float] = field(default=None, init=False, repr=False)

    def set_previous_step(self, alpha_used: float) -> None:
        """Call after line search with the step that was actually used."""
        self._prev_step = alpha_used

    def _compute_quadratic_interpolation_step(
        self,
        f0: float,
        g0: float,
        alpha_prev: float,
        f_prev: float,
    ) -> Optional[float]:
        """Quadratic f(a)=f0+g0*a+a2*a^2 through (0,f0,g0) and (alpha_prev, f_prev)."""
        if alpha_prev <= 0 or not np.isfinite(alpha_prev):
            return None
        denom = alpha_prev * alpha_prev
        if denom < 1e-20:
            return None
        a2 = (f_prev - f0 - g0 * alpha_prev) / denom
        if not np.isfinite(a2) or a2 <= 0:
            return None
        a_min = -g0 / (2.0 * a2)
        if not np.isfinite(a_min) or a_min <= 0:
            return None
        if a_min > self.max_step:
            return self.max_step
        return float(a_min)

    def estimate_step_size(self, x, objective_function) -> float:
        """Estimate step using quadratic interpolation from previous (alpha_prev, f_prev)."""
        dphi = objective_function.dphi
        if dphi is None:
            step = self._prev_step if self._prev_step is not None else self.init_step
            self._prev_f = getattr(objective_function, "phi", None)
            return max(step, self.min_step)
        f_current = getattr(objective_function, "phi", None)
        if f_current is None:
            f_current = objective_function.evaluate(x)
        f_current = float(f_current.compute()) if hasattr(f_current, "compute") else float(
            np.asarray(f_current).item()
        )
        g0 = -_vdot_real(dphi, dphi)
        if self._prev_step is None or self._prev_f is None:
            step = self.init_step
        else:
            alpha_q = self._compute_quadratic_interpolation_step(
                f_current, g0, self._prev_step, self._prev_f
            )
            if alpha_q is not None and alpha_q > self.min_step:
                step = alpha_q
            else:
                step = self._prev_step if self._prev_step is not None else self.init_step
        step = max(min(step, self.max_step), self.min_step)
        self._prev_f = f_current
        return step
