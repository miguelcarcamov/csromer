"""Barzilai-Borwein alternating: even iterations BB1, odd BB2 (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BarzilaiBorwein


@dataclass(init=True, repr=True)
class BarzilaiBorweinAlternating(BarzilaiBorwein):
    """Alternating: even k -> BB1, odd k -> BB2."""

    def _select_step_candidate(
        self, alpha_bb1: float, alpha_bb2: float, iteration: int
    ) -> Optional[float]:
        bb1_valid = np.isfinite(alpha_bb1) and alpha_bb1 > 0
        bb2_valid = np.isfinite(alpha_bb2) and alpha_bb2 > 0
        if not (bb1_valid or bb2_valid):
            return None
        if (iteration % 2) == 0:
            return alpha_bb1 if bb1_valid else (alpha_bb2 if bb2_valid else None)
        return alpha_bb2 if bb2_valid else (alpha_bb1 if bb1_valid else None)
