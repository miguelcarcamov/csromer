"""
L1 regularization term for sparse Faraday depth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...utils.array_utils import math_module
from ..fi import Fi


def _approx_abs(x, epsilon: float, xp=None):
    """
    Approximate magnitude: real or complex (Faraday depth treated as complex).

    Private helper function. Handles both numpy and dask arrays, and both real
    and complex inputs. Uses epsilon to avoid division by zero in gradients.

    Args:
        x: Input array (real or complex)
        epsilon: Small value to avoid division by zero
        xp: Math module (numpy or dask.array, default: numpy)

    Returns:
        Magnitude array (same shape as x)
    """
    if xp is None:
        xp = np
    if xp is np and hasattr(x, "compute"):
        x = x  # keep dask; iscomplexobj may need compute - use dask version
    if xp is np and hasattr(x, "__array__") and not hasattr(x, "compute"):
        x = np.asarray(x)
    if xp is np and np.iscomplexobj(x):
        return xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + epsilon)
    if xp is np:
        return xp.sqrt(x * x + epsilon)
    # dask path: support complex (use np.issubdtype for dtype check)
    if np.issubdtype(x.dtype, np.complexfloating):
        return xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + epsilon)
    return xp.sqrt(x * x + epsilon)


@dataclass(init=True, repr=True)
class L1(Fi):
    """
    L1 regularization term: sum(|x|) for sparse reconstruction.

    Non-differentiable term that promotes sparsity. Uses soft-thresholding for
    proximal operator. Supports both real and complex arrays (magnitude-based).

    Attributes:
        is_differentiable: Always False (L1 is non-differentiable at zero)
    """
    is_differentiable: bool = False

    def __post_init__(self):
        """
        Post-initialization: call parent.
        """
        super().__post_init__()

    def evaluate(self, x, epsilon: float = np.finfo(np.float32).tiny):
        """
        Evaluate L1 norm: sum(|x|).

        Public method. Computes magnitude (with epsilon smoothing) and sums.

        Args:
            x: Input array (real or complex)
            epsilon: Small value for numerical stability (default: float32 tiny)

        Returns:
            L1 norm (scalar)
        """
        xp = math_module(x)
        mag = _approx_abs(x, epsilon, xp=xp)
        result = xp.sum(mag)
        self._func_value = result
        return result

    def calculate_gradient(self, x, epsilon: float = np.finfo(np.float32).tiny):
        """
        Calculate subgradient: x / |x| (with epsilon smoothing).

        Public method. Returns subgradient (not true gradient since L1 is non-differentiable).
        Uses epsilon to avoid division by zero.

        Args:
            x: Input array (real or complex)
            epsilon: Small value for numerical stability (default: float32 tiny)

        Returns:
            Subgradient array (same shape as x)
        """
        xp = math_module(x)
        mag = _approx_abs(x, epsilon, xp=xp)
        g = x / mag
        self._grad_value = g
        return g

    def calculate_prox(self, x, nu: float = 0):
        """
        Soft-thresholding proximal operator.

        Public method. Applies soft-thresholding on magnitude (real or complex).
        When nu > 0, threshold = self.reg * nu.

        Args:
            x: Input array (real or complex)
            nu: Step size parameter (default: 0, uses self.reg as threshold)

        Returns:
            Soft-thresholded array (same shape and dtype as x)
        """
        xp = math_module(x)
        thresh = self.reg if nu == 0 else self.reg * nu
        mag = xp.abs(x)
        eps = np.finfo(np.float32).tiny
        scale = xp.maximum(1.0 - thresh / (mag + eps), 0.0)
        return (x * scale).astype(x.dtype)
