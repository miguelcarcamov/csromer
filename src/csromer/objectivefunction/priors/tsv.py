"""
Total Squared Variation (TSV) regularization term for smooth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import prox_tv as ptv

from ..fi import Fi


@dataclass(init=True, repr=True)
class TSV(Fi):
    """
    Total Squared Variation (TSV) regularization: sum(|x[i+1] - x[i]|^2).

    Differentiable term that promotes smoothness (squared differences). Uses prox_tv
    library for efficient proximal operator (TV2).

    Attributes:
        is_differentiable: Always True (TSV is differentiable)
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
        Evaluate TSV norm: sum(|x[i+1] - x[i]|^2).

        Public method. Computes sum of squared differences between adjacent elements.

        Args:
            x: Input array (1D)

        Returns:
            TSV norm (scalar)
        """
        tv = 0.0
        n = x.shape[0]
        for i in range(0, n - 1):
            tv += np.abs(x[i + 1] - x[i])**2
        return tv

    def calculate_gradient(self, x) -> np.ndarray:
        """
        Calculate gradient of TSV.

        Public method. Computes gradient of squared differences.

        Args:
            x: Input array (1D)

        Returns:
            Gradient array (same shape as x)
        """
        n = len(x)
        dx = np.zeros(n, dtype=x.dtype)
        for i in range(1, n - 1):
            dx[i] = 2.0 * (np.sign(x[i] - x[i - 1]) - np.sign(x[i + 1] - x[i]))
        return dx

    def calculate_prox(self, x, nu: float = 0.0) -> np.ndarray:
        """
        Proximal operator: TSV denoising via prox_tv (TV2).

        Public method. Uses prox_tv library for efficient TSV proximal operator.

        Args:
            x: Input array (1D)
            nu: Step size parameter (not used, threshold is self.reg)

        Returns:
            TSV-denoised array (same shape as x)
        """
        return ptv.tv2_1d(x, self.reg)
