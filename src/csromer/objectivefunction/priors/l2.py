"""
L2 regularization term for Faraday depth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fi import Fi


def _l2_norm(x: np.ndarray) -> float:
    """
    Compute L2 norm: sqrt(sum(x^2)).

    Private helper function.

    Args:
        x: Input array

    Returns:
        L2 norm (scalar)
    """
    return np.sqrt(np.sum(x**2))


def _approx_l2_norm(x: np.ndarray, epsilon: float) -> float:
    """
    Approximate L2 norm with epsilon smoothing.

    Private helper function.

    Args:
        x: Input array
        epsilon: Small value for numerical stability

    Returns:
        Approximate L2 norm (scalar)
    """
    return _l2_norm(x + epsilon)


@dataclass(init=True, repr=True)
class L2(Fi):
    """
    L2 regularization term: ||x||_2.

    Differentiable term that promotes smoothness. Uses epsilon smoothing for
    numerical stability in gradients.

    Attributes:
        is_differentiable: Always True (L2 is differentiable)
    """

    def __post_init__(self):
        """
        Post-initialization: call parent.
        """
        super().__post_init__()

    def evaluate(self, x, epsilon: float = np.finfo(np.float32).tiny) -> float:
        """
        Evaluate L2 norm: ||x||_2.

        Public method. Uses epsilon smoothing for numerical stability.

        Args:
            x: Input array
            epsilon: Small value for numerical stability (default: float32 tiny)

        Returns:
            L2 norm (scalar)
        """
        val = _approx_l2_norm(x, epsilon)
        return val

    def calculate_gradient(self, x, epsilon: float = np.finfo(np.float32).tiny) -> np.ndarray:
        """
        Calculate gradient: x / ||x||_2.

        Public method. Uses epsilon smoothing to avoid division by zero.

        Args:
            x: Input array
            epsilon: Small value for numerical stability (default: float32 tiny)

        Returns:
            Gradient array (same shape as x)
        """
        dx = np.zeros(len(x), dtype=x.dtype)
        dx = x / _approx_l2_norm(x, epsilon)
        return dx

    def calculate_prox(self, x, nu: float = 0) -> np.ndarray:
        """
        Proximal operator: soft-thresholding on L2 norm.

        Public method. Projects onto L2 ball when threshold > 0.

        Args:
            x: Input array
            nu: Step size parameter (not used, threshold is self.reg)

        Returns:
            Proximal result (same shape as x)
        """
        l2_factor = 1.0 - (self.reg / _l2_norm(x))
        l2_prox = np.maximum(l2_factor, 0.0)
        print(l2_prox)
        return x * l2_prox
