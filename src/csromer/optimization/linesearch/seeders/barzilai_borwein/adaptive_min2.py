"""ABBmin2: use previous step when alpha_bb2/alpha_bb1 < tau (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BarzilaiBorwein


@dataclass(init=True, repr=True)
class BarzilaiBorweinAdaptiveMin2(BarzilaiBorwein):
    """ABBmin2: when alpha_bb2/alpha_bb1 < tau use previous step, else BB1."""

    tau: float = 0.9

    def _select_step_candidate(self, alpha_bb1: float, alpha_bb2: float,
                               iteration: int) -> Optional[float]:
        if (not np.isfinite(alpha_bb1)) or (alpha_bb1 <= 0.0):
            return self._prev_step if self._prev_step is not None else self.init_step
        if np.isfinite(alpha_bb2) and alpha_bb2 > 0 and (alpha_bb2 / alpha_bb1 < self.tau):
            if self._prev_step is not None:
                return self._prev_step
        return alpha_bb1
