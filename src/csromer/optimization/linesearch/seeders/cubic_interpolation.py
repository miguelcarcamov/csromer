"""Cubic interpolation step seeder (Pyralysis-style). Uses (0, f, g) and (alpha_prev, f_prev)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ....utils.array_utils import maybe_compute
from .base import StepSizeSeeder


def _vdot_real(a, b) -> float:
    out = np.real(np.vdot(np.ravel(a), np.ravel(b)))
    return float(maybe_compute(out))


@dataclass(init=True, repr=True)
class CubicInterpolationSeeder(StepSizeSeeder):
    """
    Cubic interpolation along search direction.
    Fits (0, f_current, g_current) and (alpha_prev, f_prev); caller must call
    set_previous_step(alpha_used) after each line search.
    """

    max_step: float = 10.0
    _prev_f: Optional[float] = field(default=None, init=False, repr=False)
    _prev_step: Optional[float] = field(default=None, init=False, repr=False)

    def set_previous_step(self, alpha_used: float) -> None:
        """Call after line search with the step that was actually used."""
        self._prev_step = alpha_used

    def _find_cubic_minimum(
        self,
        f0: float,
        g0: float,
        alpha_prev: float,
        f_prev: float,
    ) -> Optional[float]:
        """Cubic through (0, f0, g0) and (alpha_prev, f_prev). Find minimizer in (0, alpha_prev]."""
        # f(a) = f0 + g0*a + a2*a^2 + a3*a^3, f'(a) = g0 + 2*a2*a + 3*a3*a^2
        # f(alpha_prev)=f_prev => a2*ap^2 + a3*ap^3 = f_prev - f0 - g0*ap  (1)
        # We need f'(alpha_prev) for cubic - standard is to use two points + two derivatives.
        # With only (0,f0,g0) and (ap, f_prev) we have 2 dof; use quadratic or get g_prev.
        # Pyralysis-style cubic often uses (0,f0,g0) and (ap, f_prev, g_prev). Here we don't have g_prev.
        # So use quadratic: f(a)=f0+g0*a+a2*a^2. f(ap)=f_prev => a2 = (f_prev-f0-g0*ap)/ap^2.
        # Minimizer: f'(a)=g0+2*a2*a=0 => a = -g0/(2*a2) if a2>0.
        if alpha_prev <= 0 or not np.isfinite(alpha_prev):
            return None
        denom = alpha_prev * alpha_prev
        if denom < 1e-20:
            return None
        a2 = (f_prev - f0 - g0 * alpha_prev) / denom
        if not np.isfinite(a2):
            return None
        if a2 <= 0:
            return None
        a_min = -g0 / (2.0 * a2)
        if not np.isfinite(a_min) or a_min <= 0:
            return None
        if a_min > self.max_step:
            return self.max_step
        return float(a_min)

    def estimate_step_size(self, x, objective_function) -> float:
        """Estimate step using cubic/quadratic interpolation from previous (alpha_prev, f_prev)."""
        # Current: (0, f_current, g_current); direction = -dphi => g0 = -||dphi||^2
        dphi = objective_function.dphi
        if dphi is None:
            step = self._prev_step if self._prev_step is not None else self.init_step
            self._prev_f = getattr(objective_function, "phi", None)
            return max(step, self.min_step)
        f_current = getattr(objective_function, "phi", None)
        if f_current is None:
            f_current = objective_function.evaluate(x)
        f_current = float(maybe_compute(f_current))
        g0 = -_vdot_real(dphi, dphi)
        if self._prev_step is None or self._prev_f is None:
            step = self.init_step
        else:
            alpha_cubic = self._find_cubic_minimum(
                f_current, g0, self._prev_step, self._prev_f
            )
            if alpha_cubic is not None and alpha_cubic > self.min_step:
                step = alpha_cubic
            else:
                step = self._prev_step if self._prev_step is not None else self.init_step
        step = max(min(step, self.max_step), self.min_step)
        self._prev_f = f_current
        return step
