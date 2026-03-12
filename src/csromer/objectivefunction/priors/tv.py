"""
Total Variation (TV) regularization term for piecewise-constant reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import prox_tv as ptv

from ..fi import Fi


@dataclass(init=True, repr=True)
class TV(Fi):
    """
    Total Variation (TV) regularization: sum(|x[i+1] - x[i]|).

    Non-differentiable term that promotes piecewise-constant solutions. Uses prox_tv
    library for efficient proximal operator.

    Attributes:
        is_differentiable: Always False (TV is non-differentiable)
        nu: Internal array (unused, kept for compatibility)
    """
    is_differentiable: bool = False
    nu: np.ndarray = field(init=False, default=np.array([]))

    def __post_init__(self):
        """
        Post-initialization: call parent.
        """
        super().__post_init__()

    def evaluate(self, x) -> float:
        """
        Evaluate TV norm: sum(|x[i+1] - x[i]|).

        Public method. Computes sum of absolute differences between adjacent elements.

        Args:
            x: Input array (1D)

        Returns:
            TV norm (scalar)
        """
        tv = 0.0
        n = x.shape[0]
        for i in range(0, n - 1):
            tv += np.abs(x[i + 1] - x[i])
        return tv

    def calculate_gradient(self, x) -> np.ndarray:
        """
        Calculate subgradient of TV.

        Public method. Returns subgradient (not true gradient since TV is non-differentiable).
        Computes sign differences at interior points.

        Args:
            x: Input array (1D)

        Returns:
            Subgradient array (same shape as x)
        """
        n = len(x)
        dx = np.zeros(n, dtype=x.dtype)
        for i in range(1, n - 1):
            dx[i] = np.sign(x[i] - x[i - 1]) - np.sign(x[i + 1] - x[i])
        return dx

    def calculate_prox(self, x, nu: float = 0.0) -> np.ndarray:
        """
        Proximal operator: TV denoising via prox_tv.

        Public method. Uses prox_tv library for efficient TV proximal operator.

        Args:
            x: Input array (1D)
            nu: Step size parameter (not used, threshold is self.reg)

        Returns:
            TV-denoised array (same shape as x)
        """
        return ptv.tv1_1d(x, self.reg)
