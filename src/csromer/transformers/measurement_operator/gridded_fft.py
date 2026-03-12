"""
Gridded FFT for uniformly spaced lambda². Uses da.fft.fft / da.fft.ifft
when inputs are dask arrays.

Implements the Faraday depth Fourier transform (Burn 1966):
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

No l2_ref in the transform; gridded lambda² should be 0, d_l2, 2*d_l2, ...
so FFT bins align with physical channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from ...utils.array_utils import asnumpy, is_dask_array
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

    No l2_ref in forward/adjoint. Expects lambda² grid 0, d_l2, ..., (n-1)*d_l2.
    Uses da.fft.fft/ifft for dask arrays to maintain lazy computation.
    """

    def __post_init__(self):
        """Post-initialization: call configure if dataset and parameter are available."""
        super().__post_init__()
        if self.dataset is not None and self.parameter is not None:
            self.configure()

    def configure(self) -> None:
        """
        Configure operator (no phase factor; transform does not use l2_ref).
        """
        if self.dataset is None or self.parameter is None:
            return
        phi = asnumpy(self.parameter.phi)
        # No l2_ref in transform: phase = 1
        self._l2_ref_phase = np.ones(phi.shape[0], dtype=np.complex64)

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
        phase_conj = np.conj(self._l2_ref_phase)  # = 1, no l2_ref
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

        Public method. For gridded FFT, RMTF is uniform (all ones).

        Args:
            phi_x: Faraday depth of point source (rad/m², unused for gridded FFT)

        Returns:
            RMTF array (n_phi,) of ones
        """
        n = self.parameter.phi.shape[0]
        if da is not None:
            return da.ones(n, dtype=np.complex64, chunks=(n,))
        return np.ones(n, dtype=np.complex64)
