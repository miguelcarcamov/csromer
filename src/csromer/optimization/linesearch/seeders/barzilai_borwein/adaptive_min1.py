"""ABBmin1: use min of BB2 history when alpha_bb2/alpha_bb1 < tau (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import BarzilaiBorwein


@dataclass(init=True, repr=True)
class BarzilaiBorweinAdaptiveMin1(BarzilaiBorwein):
    """ABBmin1: when alpha_bb2/alpha_bb1 < tau use min(BB2 history), else BB1."""

    tau: float = 0.8
    window: int = 10
    bb2_history: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32), init=False, repr=False
    )
    _bb2_write_index: int = field(default=0, init=False, repr=False)

    def _select_step_candidate(self, alpha_bb1: float, alpha_bb2: float,
                               iteration: int) -> Optional[float]:
        bb1_valid = alpha_bb1 is not None and np.isfinite(alpha_bb1) and alpha_bb1 > 0.0
        bb2_valid = alpha_bb2 is not None and np.isfinite(alpha_bb2) and alpha_bb2 > 0.0
        if bb2_valid:
            if len(self.bb2_history) < self.window:
                self.bb2_history = np.append(self.bb2_history, alpha_bb2)
            else:
                self.bb2_history[self._bb2_write_index] = alpha_bb2
                self._bb2_write_index = (self._bb2_write_index + 1) % self.window
        if not (bb1_valid and bb2_valid):
            return None
        ratio = alpha_bb2 / alpha_bb1
        if np.isfinite(ratio) and ratio < self.tau:
            return float(np.min(self.bb2_history))
        return alpha_bb1
