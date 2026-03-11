"""
Direct Fourier transform (non-gridded lambda²): forward and adjoint
with dask-aware implementation (no unnecessary numpy conversion).

Implements the Faraday depth Fourier transform (Burn 1966):
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

Uses kernel exp(+2j * phi * lambda²) with no reference lambda² in the
transform; l2_ref is not used in forward/adjoint (consistent with "full resolution").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from ...utils.array_utils import asnumpy, is_dask_array, maybe_compute
from .base import MeasurementOperator

try:
    import dask.array as da
except ImportError:
    da = None


def _exp_forward(l2, phi):
    """
    Compute forward exponential: exp(+2j * phi[:, None] * l2[None, :]).

    Private helper. Kernel exp(+2j * phi * lambda²) for Faraday depth Fourier transform.
    Returns shape (n_phi, n_channels). Handles both numpy and dask arrays.

    Args:
        l2: Lambda² at each channel (n_channels,)
        phi: Faraday depth grid (n_phi,)

    Returns:
        Exponential array (n_phi, n_channels)
    """
    if da is not None and is_dask_array(l2):
        phi_2d = np.asarray(phi)[:, np.newaxis]
        l2_2d = l2[np.newaxis, :]
        return da.exp(2.0j * phi_2d * l2_2d)
    l2_np = np.asarray(l2)
    phi_a = np.asarray(phi)
    return np.exp(2.0j * phi_a[:, np.newaxis] * l2_np[np.newaxis, :]).astype(np.complex64)


def _exp_adjoint(l2, phi):
    """
    Compute adjoint exponential: exp(-2j * l2[:, None] * phi[None, :]).

    Private helper. Adjoint kernel exp(-2j * lambda² * phi), conjugate transpose of forward.
    Returns shape (n_channels, n_phi). Handles both numpy and dask arrays.

    Args:
        l2: Lambda² at each channel (n_channels,)
        phi: Faraday depth grid (n_phi,)

    Returns:
        Exponential array (n_channels, n_phi)
    """
    if da is not None and is_dask_array(l2):
        l2_2d = l2[:, np.newaxis]
        phi_2d = np.asarray(phi)[np.newaxis, :]
        return da.exp(-2.0j * l2_2d * phi_2d)
    l2_np = np.asarray(l2)
    phi_a = np.asarray(phi)
    return np.exp(-2.0j * l2_np[:, np.newaxis] * phi_a[np.newaxis, :]).astype(np.complex64)


@dataclass(init=True, repr=True)
class DirectFourier1D(MeasurementOperator):
    """
    Direct Fourier transform for non-gridded lambda².

    Implements the Faraday depth Fourier transform (Burn 1966):
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

    Uses kernel exp(+2j * phi * lambda²); no l2_ref in forward/adjoint.
    forward(x): Faraday depth -> data (lambda²)
    adjoint(b): data -> Faraday depth

    Dask-aware: uses da.einsum / da.exp when inputs are dask arrays.
    """

    def configure(self) -> None:
        """
        Configure operator (no-op for DirectFourier1D).

        Public method. Override in subclasses if configuration is needed.
        """
        pass

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²).

        Protected method. Computes exp(+2j * phi * lambda²) and applies via einsum/dot.
        No l2_ref in the transform (Burn / full-resolution convention).

        Args:
            x: Complex Faraday depth spectrum (n_phi,)

        Returns:
            Complex polarization P(lambda²) (n_channels,)
        """
        l2 = self.dataset.lambda2
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        exp_f = _exp_forward(l2, phi)
        if da is not None and is_dask_array(exp_f):
            out = da.einsum("i,ij->j", x, exp_f)
            return out.astype(np.complex64)
        return np.dot(np.asarray(x), exp_f).astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator implementation: P(lambda²) -> phi.

        Protected method. Computes exp(-2j * lambda² * phi) and applies via einsum/dot.
        No l2_ref in the transform. Raw adjoint: A^H(b), no weights, no K.

        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (unused)

        Returns:
            Complex Faraday depth spectrum (n_phi,)
        """
        l2 = self.dataset.lambda2
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        exp_adj = _exp_adjoint(l2, phi)
        if da is not None and is_dask_array(b):
            x = da.einsum("j,ji->i", b, exp_adj)
            return x.astype(np.complex64)
        return np.dot(np.asarray(b), np.asarray(exp_adj)).astype(np.complex64)

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        """
        Rotation Measure Transfer Function (RMTF).

        Public method. Computes response to a point source at phi_x. Uses weighted
        adjoint of ones.

        Args:
            phi_x: Faraday depth of point source (rad/m², default: 0.0)

        Returns:
            RMTF array (n_phi,)
        """
        l2 = self.dataset.lambda2
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        w = self.dataset.w
        s = self.dataset.s if self.dataset.s is not None else (da.ones_like(w) if (da and is_dask_array(w)) else np.ones_like(w))
        k = float(maybe_compute(self.dataset.k)) if self.dataset.k is not None else 1.0
        weights = w / s
        exp_adj = _exp_adjoint(l2, phi)
        if da is not None and (is_dask_array(weights) or is_dask_array(exp_adj)):
            x = da.einsum("j,ji->i", weights, exp_adj)
            return (x / k).astype(np.complex64)
        return (np.dot(np.asarray(weights), np.asarray(exp_adj)) / k).astype(np.complex64)
