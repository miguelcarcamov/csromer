"""
Direct Fourier transform (non-gridded lambda²): forward and adjoint
with dask-aware implementation (no unnecessary numpy conversion).
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


def _exp_forward(l2_diff, phi):
    """exp(2j * phi[:, None] * l2_diff[None, :]) -> shape (n_phi, n_channels)."""
    if da is not None and is_dask_array(l2_diff):
        phi_2d = np.asarray(phi)[:, np.newaxis]
        l2_2d = l2_diff[np.newaxis, :]
        return da.exp(2.0j * phi_2d * l2_2d)
    l2 = np.asarray(l2_diff)
    phi_a = np.asarray(phi)
    return np.exp(2.0j * phi_a[:, np.newaxis] * l2[np.newaxis, :]).astype(np.complex64)


def _exp_adjoint(l2_diff, phi):
    """exp(-2j * l2_diff[:, None] * phi[None, :]) -> shape (n_channels, n_phi)."""
    if da is not None and is_dask_array(l2_diff):
        l2_2d = l2_diff[:, np.newaxis]
        phi_2d = np.asarray(phi)[np.newaxis, :]
        return da.exp(-2.0j * l2_2d * phi_2d)
    l2 = np.asarray(l2_diff)
    phi_a = np.asarray(phi)
    return np.exp(-2.0j * l2[:, np.newaxis] * phi_a[np.newaxis, :]).astype(np.complex64)


@dataclass(init=True, repr=True)
class DirectFourier1D(MeasurementOperator):
    """
    Direct Fourier transform for non-gridded lambda².
    forward(x): Faraday depth -> data (lambda²). adjoint(b): data -> Faraday depth.
    Dask-aware: uses da.einsum / da.exp when inputs are dask arrays.
    """

    def configure(self) -> None:
        pass

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        # x: (n_phi,) complex, result: (n_channels)
        l2_diff = self.dataset.lambda2 - self.dataset.l2_ref
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        exp_f = _exp_forward(l2_diff, phi)
        if da is not None and is_dask_array(exp_f):
            out = da.einsum("i,ij->j", x, exp_f)
            return out.astype(np.complex64)
        return np.dot(np.asarray(x), exp_f).astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        # Raw adjoint: A^H(b), no weights, no K. Caller passes weighted residuals for gradients.
        l2_diff = self.dataset.lambda2 - self.dataset.l2_ref
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        exp_adj = _exp_adjoint(l2_diff, phi)
        if da is not None and is_dask_array(b):
            x = da.einsum("j,ji->i", b, exp_adj)
            return x.astype(np.complex64)
        return np.dot(np.asarray(b), np.asarray(exp_adj)).astype(np.complex64)

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        # Response: adjoint of ones (weighted)
        l2_diff = self.dataset.lambda2 - self.dataset.l2_ref
        phi = self.parameter.phi
        if da is not None and is_dask_array(phi):
            phi = asnumpy(phi)
        w = self.dataset.w
        s = self.dataset.s if self.dataset.s is not None else (da.ones_like(w) if (da and is_dask_array(w)) else np.ones_like(w))
        k = float(maybe_compute(self.dataset.k)) if self.dataset.k is not None else 1.0
        weights = w / s
        exp_adj = _exp_adjoint(l2_diff, phi)
        if da is not None and (is_dask_array(weights) or is_dask_array(exp_adj)):
            x = da.einsum("j,ji->i", weights, exp_adj)
            return (x / k).astype(np.complex64)
        return (np.dot(np.asarray(weights), np.asarray(exp_adj)) / k).astype(np.complex64)
