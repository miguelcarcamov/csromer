"""
Gridded FFT for uniformly spaced lambda². Uses da.fft.fft / da.fft.ifft
when inputs are dask arrays.

Implements the Faraday depth Fourier transform (Burn 1966):
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

Gridded lambda² is l2_0, l2_0+d_l2, ..., l2_0+(n-1)*d_l2 (with l2_0 = dataset.l2_min > 0).
The FFT implements the kernel at these physical λ²; the phase exp(2j φ l2_0) is applied
so that when l2_0 = 0 (full resolution, grid at 0) the phase is 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from ...utils.array_utils import asnumpy, is_dask_array, math_module
from .base import MeasurementOperator

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(init=True, repr=True)
class GriddedFFT1D(MeasurementOperator):
    """
    FFT-based measurement operator when lambda² is on a regular grid.

    Implements the Faraday depth Fourier transform (Burn 1966):
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

    Expects gridded lambda² = l2_0, l2_0+d_l2, ... (l2_0 = first channel, typically > 0).
    Applies phase exp(2j φ l2_0) so the kernel is correct for physical λ²; when l2_0=0
    (e.g. full resolution with grid at 0) the phase is 1.
    """

    def __post_init__(self):
        """Post-initialization: call configure if dataset and parameter are available."""
        super().__post_init__()
        if self.dataset is not None and self.parameter is not None:
            self.configure()

    def configure(self) -> None:
        """
        Set phase from first gridded λ² (l2_0). Kernel is exp(2j φ λ²_k) with
        λ²_k = l2_0 + k*d_l2, so we phase F by exp(2j φ l2_0) before the FFT.
        """
        if self.dataset is None or self.parameter is None:
            return
        phi = asnumpy(self.parameter.phi)
        l2_grid = asnumpy(self.dataset.lambda2)
        l2_0 = float(l2_grid[0])
        self._l2_ref_phase = np.exp(2j * phi * l2_0).astype(np.complex64)

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²) via FFT.

        Protected method. No l2_ref; shift then ifft for positive sign convention.
        """
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        # Phase is 1 (no l2_ref); x_phased = x
        x_phased = x * self._l2_ref_phase
        # ifftshift so phi=0 maps to FFT DC; ifft(norm="forward") for exp(+2j*phi*lambda²)
        if da is not None and is_dask_array(x_phased):
            x_shifted = da.fft.ifftshift(x_phased)
            return da.fft.ifft(x_shifted, norm="forward").astype(np.complex64)
        x_shifted = np.fft.ifftshift(x_phased)
        return np.fft.ifft(x_shifted, norm="forward").astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator implementation: P(lambda²) -> phi via IFFT.
        No l2_ref; FFT then fftshift (adjoint of forward).
        """
        if da is not None and is_dask_array(b):
            x_fft = da.fft.fft(b, norm="forward").astype(np.complex64)
            x_fft = da.fft.fftshift(x_fft).astype(np.complex64)
        else:
            x_fft = np.fft.fft(b, norm="forward").astype(np.complex64)
            x_fft = np.fft.fftshift(x_fft).astype(np.complex64)
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        phase_conj = np.conj(self._l2_ref_phase)
        if da is not None and is_dask_array(x_fft):
            return (x_fft * phase_conj).astype(np.complex64)
        return (x_fft * phase_conj).astype(np.complex64)

    def _dirty_spectrum_impl(self, data: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Dirty spectrum: A^H(weighted data) with scaling so the result matches the
        continuous definition (sum over channels; no 1/N from FFT).
        The FFT adjoint with norm="forward" yields (1/N)*sum; multiply by N here.
        """
        raw = super()._dirty_spectrum_impl(data)
        n_chan = self.dataset.m
        return raw * n_chan

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        """
        Rotation Measure Transfer Function (RMTF).

        Adjoint of (weights / sum(weights)), then multiplied by n_chan. Uses
        this class's adjoint (IFFT-based) so the gridded transform is used.

        Args:
            phi_x: Faraday depth of point source (rad/m², unused)

        Returns:
            RMTF array (n_phi,), same units as dirty map.
        """
        if self.dataset is None:
            raise RuntimeError("dataset is required for RMTF")
        w = self.dataset.w
        s = getattr(self.dataset, "s", None)
        xp = math_module(w)
        weights = (w / s) if s is not None else w
        sum_w = xp.sum(weights)
        sum_w = sum_w.compute() if hasattr(sum_w, "compute") else sum_w
        sum_w = float(sum_w) if sum_w is not None else 1.0
        if abs(sum_w) < 1e-10:
            sum_w = 1.0
        normalized = weights / sum_w
        # Use this class's adjoint (IFFT path), not the base helper
        raw = self.adjoint(normalized)
        l2_ref = getattr(self.dataset, "l2_ref", None)
        if (
            self.parameter is not None
            and l2_ref is not None
            and abs(float(l2_ref)) >= 1e-10
        ):
            xp = math_module(raw)
            phi = self.parameter.phi
            phi_same = xp.asarray(phi)
            phase_ramp = xp.exp(2.0j * phi_same * float(l2_ref)).astype(np.complex64)
            raw = raw * phase_ramp
        n_chan = self.dataset.m
        return (raw * n_chan).astype(np.complex64)
