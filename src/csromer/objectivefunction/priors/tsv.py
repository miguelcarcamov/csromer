"""
Total Squared Variation (TSV) regularization term for smooth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import prox_tv as ptv

from ...utils.array_utils import math_module
from ..fi import Fi


@dataclass(init=True, repr=True)
class TSV(Fi):
    """
    Total Squared Variation (TSV) regularization: sum(|x[i+1] - x[i]|^2).

    Differentiable term that promotes smooth (as opposed to piecewise-constant)
    solutions. Unlike TV, TSV admits no isotropic/anisotropic distinction: squaring
    the modulus makes the two definitions identical, since
    |d|^2 == Re(d)^2 + Im(d)^2 exactly. It is therefore invariant under a global
    phase rotation without needing a variant flag.

    Attributes:
        is_differentiable: Always True (TSV is a smooth quadratic in the differences)
    """
    is_differentiable: bool = True

    def __post_init__(self):
        """
        Post-initialization: call parent.
        """
        super().__post_init__()

    def evaluate(self, x):
        """
        Evaluate TSV: sum(|x[i+1] - x[i]|^2).

        Public method. Vectorized and dask-compatible; works for real and complex
        input, where |.| is the complex modulus.

        Args:
            x: Input array (1D, real or complex)

        Returns:
            TSV value (scalar)
        """
        xp = math_module(x)
        d = xp.diff(x)
        result = xp.sum(xp.abs(d)**2)
        self._func_value = result
        return result

    def calculate_gradient(self, x):
        """
        Calculate gradient of TSV: 2 * D^T D x.

        Public method. TSV is smooth, so this is a true gradient, not a subgradient.
        For f = sum_i |x[i+1] - x[i]|^2 the interior entries are
        2 * ((x[i] - x[i-1]) - (x[i+1] - x[i])), with the endpoints carrying only the
        single difference each participates in. Vectorized and dask-compatible.

        Args:
            x: Input array (1D, real or complex)

        Returns:
            Gradient array (same shape and dtype as x)
        """
        xp = math_module(x)
        d = 2.0 * xp.diff(x)
        zero = xp.zeros(1, dtype=x.dtype)
        # d[i] contributes -2*d[i] to entry i and +2*d[i] to entry i+1, which yields
        # the correct endpoint terms without special-casing them.
        grad = xp.concatenate([zero, d]) - xp.concatenate([d, zero])
        self._grad_value = grad
        return grad

    def calculate_prox(self, x, nu: float = 0.0):
        """
        Proximal operator for TSV.

        Public method. Solves min_z 0.5*||z - x||^2 + reg*||D z||^2, whose solution is
        the linear system (I + 2*reg*D^T D) z = x. Delegates to prox_tv for now.

        Note: prox_tv's tv2_1d minimizes ||D z||_2 (the norm) rather than ||D z||^2
        (its square), so this does not currently match evaluate(). Replacing it with an
        exact tridiagonal solve is tracked in #27. Complex input is handled
        channel-wise because prox_tv is real-valued.

        Args:
            x: Input array (1D, real or complex)
            nu: Step size. The effective threshold is reg*nu, or reg when nu == 0.

        Returns:
            Denoised array (same shape and dtype as x)
        """
        reg = self.reg if nu == 0 else self.reg * nu
        x_np = np.asarray(x)
        if np.iscomplexobj(x_np):
            real = ptv.tv2_1d(np.ascontiguousarray(x_np.real, dtype=np.float64), reg)
            imag = ptv.tv2_1d(np.ascontiguousarray(x_np.imag, dtype=np.float64), reg)
            return (real + 1j * imag).astype(x_np.dtype)
        return ptv.tv2_1d(np.ascontiguousarray(x_np, dtype=np.float64), reg).astype(x_np.dtype)
