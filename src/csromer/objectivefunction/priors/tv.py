"""
Total Variation (TV) regularization term for piecewise-constant reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import prox_tv as ptv

from ...utils.array_utils import math_module
from ..fi import Fi


@dataclass(init=True, repr=True)
class TV(Fi):
    """
    Total Variation (TV) regularization: sum(|x[i+1] - x[i]|).

    Non-differentiable term that promotes piecewise-constant solutions.

    For complex input this is the isotropic form, |d| being the complex modulus,
    which is invariant under a global phase rotation. The anisotropic alternative,
    sum(|Re d| + |Im d|), is not: it penalizes some absolute polarization angles up
    to sqrt(2) more than others, and that bias is fully realized when the differences
    are phase-coherent — precisely the case for a well-reconstructed Faraday-thin
    component. Exposing both variants explicitly is tracked in #27.

    Attributes:
        is_differentiable: Always False (TV is non-differentiable at zero)
        nu: Internal array (unused, kept for compatibility)
    """
    is_differentiable: bool = False
    nu: np.ndarray = field(init=False, default=np.array([]))

    def __post_init__(self):
        """
        Post-initialization: call parent.
        """
        super().__post_init__()

    def evaluate(self, x):
        """
        Evaluate TV: sum(|x[i+1] - x[i]|).

        Public method. Vectorized and dask-compatible; works for real and complex
        input, where |.| is the complex modulus (isotropic TV).

        Args:
            x: Input array (1D, real or complex)

        Returns:
            TV value (scalar)
        """
        xp = math_module(x)
        result = xp.sum(xp.abs(xp.diff(x)))
        self._func_value = result
        return result

    def calculate_gradient(self, x, epsilon: float = np.finfo(np.float32).eps):
        """
        Calculate a subgradient of TV.

        Public method. TV is non-differentiable where a difference vanishes, so this
        is a subgradient, not a gradient. The subgradient of |d| with respect to d is
        d/|d|, which reduces to sign(d) for real input; epsilon guards the zero case.
        Vectorized and dask-compatible.

        Args:
            x: Input array (1D, real or complex)
            epsilon: Guard against division by zero at a vanishing difference

        Returns:
            Subgradient array (same shape and dtype as x)
        """
        xp = math_module(x)
        d = xp.diff(x)
        u = d / (xp.abs(d) + epsilon)
        zero = xp.zeros(1, dtype=x.dtype)
        # d[i] = x[i+1] - x[i] contributes -u[i] to entry i and +u[i] to entry i+1,
        # which gives the correct endpoint terms without special-casing them.
        grad = xp.concatenate([zero, u]) - xp.concatenate([u, zero])
        self._grad_value = grad
        return grad

    def calculate_prox(self, x, nu: float = 0.0):
        """
        Proximal operator: TV denoising.

        Public method. Solves min_z 0.5*||z - x||^2 + reg*sum(|D z|). Delegates to
        prox_tv for now; replacing it with an in-house Condat solver, and exposing
        isotropic and anisotropic variants explicitly, is tracked in #27.

        Note: prox_tv is real-valued, so complex input is handled channel-wise here,
        which is the *anisotropic* prox and does not match this term's isotropic
        evaluate(). That mismatch is part of what #27 resolves; channel-wise is used
        meanwhile because passing complex input to prox_tv directly is worse.

        Args:
            x: Input array (1D, real or complex)
            nu: Step size. The effective threshold is reg*nu, or reg when nu == 0.

        Returns:
            Denoised array (same shape and dtype as x)
        """
        reg = self.reg if nu == 0 else self.reg * nu
        x_np = np.asarray(x)
        if np.iscomplexobj(x_np):
            real = ptv.tv1_1d(np.ascontiguousarray(x_np.real, dtype=np.float64), reg)
            imag = ptv.tv1_1d(np.ascontiguousarray(x_np.imag, dtype=np.float64), reg)
            return (real + 1j * imag).astype(x_np.dtype)
        return ptv.tv1_1d(np.ascontiguousarray(x_np, dtype=np.float64), reg).astype(x_np.dtype)
