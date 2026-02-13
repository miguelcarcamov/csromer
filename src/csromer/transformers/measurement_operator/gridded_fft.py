"""
Gridded FFT for uniformly spaced lambda². Uses da.fft.fft / da.fft.ifft
when inputs are dask arrays.

Implements the Faraday depth Fourier transform with proper lambda²_0 phase factor:
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

For gridded lambda² uniformly spaced, applies exp(+2j * phi * lambda²_0) phase
factor before FFT to account for the reference lambda².
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


@dataclass(init=True, repr=True)
class GriddedFFT1D(MeasurementOperator):
    """
    FFT-based measurement operator when lambda² is on a regular grid.
    
    Implements the Faraday depth Fourier transform:
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi
    
    For uniformly spaced lambda² grids, applies the lambda²_0 phase factor
    exp(+2j * phi * lambda²_0) before FFT, where lambda²_0 is the reference
    lambda² (dataset.l2_ref). Uses numpy FFT convention (exp(-2πikn/N) for forward)
    with proper mapping to match the continuous FT sign convention.
    
    Uses da.fft.fft/ifft for dask arrays to maintain lazy computation.
    """

    def __post_init__(self):
        """Post-initialization: call configure if dataset and parameter are available."""
        super().__post_init__()
        if self.dataset is not None and self.parameter is not None:
            self.configure()

    def configure(self) -> None:
        """
        Configure operator: precompute lambda²_0 phase factor.
        
        Public method. Computes phase factor exp(+2j * phi * lambda²_0) for
        efficient application in forward/adjoint operations.
        """
        if self.dataset is None or self.parameter is None:
            return
        
        l2_ref = float(maybe_compute(self.dataset.l2_ref)) if self.dataset.l2_ref is not None else 0.0
        phi = asnumpy(self.parameter.phi)
        
        # Phase factor: exp(+2j * phi * lambda²_0)
        # This accounts for the reference lambda² in the gridded FFT
        self._l2_ref_phase = np.exp(2.0j * phi * l2_ref).astype(np.complex64)

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²) via FFT.
        
        Protected method. Applies lambda²_0 phase factor, then FFT to Faraday depth spectrum.
        The phase factor accounts for the reference lambda² in the gridded transform.
        
        Args:
            x: Complex Faraday depth spectrum (n_phi,)
            
        Returns:
            Complex polarization P(lambda²) (n_channels,)
        """
        # Apply lambda²_0 phase factor: exp(+2j * phi * lambda²_0)
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        
        phase = self._l2_ref_phase
        if phase is None:
            # Fallback: compute on the fly
            l2_ref = float(maybe_compute(self.dataset.l2_ref)) if self.dataset.l2_ref is not None else 0.0
            phi = asnumpy(self.parameter.phi)
            phase = np.exp(2.0j * phi * l2_ref).astype(np.complex64)
        
        # Apply phase factor before FFT
        x_phased = x * phase
        
        # For phi grid symmetric around zero: phi_n = (n - N/2) * d_phi
        # Shift input so phi=0 (at index N/2) maps to FFT DC (index 0)
        # Use ifftshift to move phi=0 to index 0
        # For Faraday depth synthesis with exp(+2j*phi*lambda²) (positive sign),
        # use ifft with norm="forward" to get exp(+2πikn/N) instead of fft's exp(-2πikn/N)
        if da is not None and is_dask_array(x_phased):
            x_shifted = da.fft.ifftshift(x_phased)
            # Use ifft with norm="forward" for positive sign convention
            return da.fft.ifft(x_shifted, norm="forward").astype(np.complex64)
        x_shifted = np.fft.ifftshift(x_phased)
        # Use ifft with norm="forward" for positive sign convention
        return np.fft.ifft(x_shifted, norm="forward").astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator implementation: P(lambda²) -> phi via IFFT.
        
        Protected method. Applies IFFT to polarization data, then removes lambda²_0 phase factor.
        The phase conjugation accounts for the reference lambda² in the gridded transform.
        
        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Complex Faraday depth spectrum (n_phi,)
        """
        # Apply FFT with norm="forward" for positive sign convention (adjoint of ifft)
        # Lambda² grid is monotonic, so no shift needed before FFT
        # Shift output to restore phi grid order (fftshift after FFT)
        # This undoes the ifftshift applied in forward direction
        if da is not None and is_dask_array(b):
            # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
            x_fft = da.fft.fft(b, norm="forward").astype(np.complex64)
            # Shift output to restore phi grid order (fftshift after FFT)
            x_fft = da.fft.fftshift(x_fft).astype(np.complex64)
        else:
            # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
            x_fft = np.fft.fft(b, norm="forward").astype(np.complex64)
            # Shift output to restore phi grid order (fftshift after FFT)
            x_fft = np.fft.fftshift(x_fft).astype(np.complex64)
        
        # Remove lambda²_0 phase factor: conjugate of exp(+2j * phi * lambda²_0)
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        
        phase = self._l2_ref_phase
        if phase is None:
            # Fallback: compute on the fly
            l2_ref = float(maybe_compute(self.dataset.l2_ref)) if self.dataset.l2_ref is not None else 0.0
            phi = asnumpy(self.parameter.phi)
            phase = np.exp(2.0j * phi * l2_ref).astype(np.complex64)
        
        # Conjugate phase factor for adjoint
        phase_conj = np.conj(phase)
        if da is not None and is_dask_array(x_fft):
            return (x_fft * phase_conj).astype(np.complex64)
        return (x_fft * phase_conj).astype(np.complex64)

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
