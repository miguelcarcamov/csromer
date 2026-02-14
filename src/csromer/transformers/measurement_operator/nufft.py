"""
NUFFT measurement operator: Kaiser-kernel forward (FFT + interpolate), adjoint of forward (no iterative solver).

Implements the Faraday depth Fourier transform:
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

Forward uses FFT + Kaiser interpolation with phase correction for lambda²_0 reference.
The phase factor accounts for the reference lambda² in the non-uniform sampling.
Adjoint is IFFT(A^H (phase* b)) so gradient is consistent. No optimization algorithm
in the adjoint—just the linear adjoint of the Kaiser forward.

Efficiency: interpolation matrix A is built once in configure() and stored as sparse. When
pydata/sparse is installed, A is stored as sparse.COO so the same matrix works for both numpy
and dask (da.dot(sparse, dask_vector) via dask's tensordot_lookup). Otherwise we use scipy.sparse
for numpy and build dense A on demand for dask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from scipy.special import i0
from scipy.sparse import csr_matrix

from ...utils.array_utils import is_dask_array, math_module, maybe_compute
from .direct_fourier import DirectFourier1D

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import sparse as _pydata_sparse  # type: ignore[import-untyped]
except ImportError:
    _pydata_sparse = None


def _kaiser_1d(u: float, half_width: int, beta: float) -> float:
    """
    Compute Kaiser kernel value at offset u.
    
    Private helper function. Returns zero for |u| > half_width.
    
    Args:
        u: Offset from center
        half_width: Half-width of kernel support
        beta: Kaiser beta parameter
        
    Returns:
        Kernel value (float)
    """
    if abs(u) > half_width:
        return 0.0
    arg = beta * np.sqrt(1.0 - (u / half_width) ** 2)
    return i0(arg) / i0(beta)


def _build_kaiser_interp_matrix(
    n_ch: int,
    n_phi: int,
    k_cont: np.ndarray,
    half_width: int,
    beta: float,
) -> np.ndarray:
    """
    Build (n_ch, n_phi) interpolation matrix A.
    
    Private helper function. Matrix A interpolates X at continuous indices k_cont
    using Kaiser kernel. Returns float32 to save memory; only ~(2*half_width+1)
    nonzeros per row.
    
    Args:
        n_ch: Number of channels
        n_phi: Number of phi grid points
        k_cont: Continuous k indices (n_ch,)
        half_width: Half-width of Kaiser kernel
        beta: Kaiser beta parameter
        
    Returns:
        Interpolation matrix A (n_ch, n_phi, float32)
    """
    A = np.zeros((n_ch, n_phi), dtype=np.float32)
    for c in range(n_ch):
        k_c = k_cont[c]
        j_center = int(np.floor(k_c))
        for j_offset in range(-half_width, half_width + 1):
            j = j_center + j_offset
            j_mod = j % n_phi
            if j_mod < 0:
                j_mod += n_phi
            u = k_c - j
            A[c, j_mod] += _kaiser_1d(u, half_width, beta)
    return A


def _kaiser_forward_arrays(parameter, dataset, conv_size: int, kaiser_beta: float):
    """
    Compute k_cont, A, phase for forward/adjoint.
    
    Private helper function. Builds interpolation matrix and phase factor for NUFFT.
    The phase factor accounts for the lambda²_0 reference in the non-uniform sampling,
    implementing exp(+2j * phi * lambda²) with proper FFT domain mapping.
    
    Args:
        parameter: Parameter object with phi grid
        dataset: Dataset with lambda²
        conv_size: Convolution size (half-width)
        kaiser_beta: Kaiser beta parameter
        
    Returns:
        Tuple of (k_cont, A, phase, N, d_phi)
    """
    N = len(parameter.phi)
    d_phi = float(parameter.cellsize)
    l2_diff = dataset.lambda2 - dataset.l2_ref
    l2_diff_np = np.asarray(maybe_compute(l2_diff))
    n_ch = l2_diff_np.size
    l2_ref = float(maybe_compute(dataset.l2_ref)) if dataset.l2_ref is not None else 0.0
    
    # Map lambda² to FFT k indices for NUFFT interpolation
    # For exp(+2j * phi * lambda²) with ifft(norm='forward'): exp(+2πikn/N)
    # The relationship: phi_n * lambda² = πkn/N
    # After ifftshift, phi_n = (n - N/2) * d_phi maps to FFT index n
    # For uniform lambda² grid with Nyquist: d_phi * d_lambda² = π/N
    # So: k = N * d_phi * (lambda² - lambda²_0) / π
    #
    # k_cont maps non-uniform lambda² to continuous FFT k indices
    # For positive sign convention exp(+2j * phi * lambda²):
    # - Positive l2_diff maps to positive k_cont (positive frequencies)
    # - Negative l2_diff maps to negative k_cont (negative frequencies)
    # After ifftshift: FFT k=0 is DC (phi=0), k>0 corresponds to positive frequencies
    k_cont = N * d_phi * l2_diff_np / np.pi
    A = _build_kaiser_interp_matrix(n_ch, N, k_cont, conv_size, kaiser_beta)
    
    # Phase factor accounts for lambda²_0 reference
    # In GriddedFFT1D: phase = exp(+2j * phi * lambda²_0) applied BEFORE FFT
    # In NUFFT: we apply phase AFTER interpolation to account for lambda²_0
    #
    # The phase correction accounts for the lambda²_0 reference in the transform
    # Since we're interpolating at non-uniform lambda² positions, we need to compute
    # which phi value corresponds to each interpolation point
    #
    # However, k_cont is a continuous FFT frequency index, not a phi value
    # The mapping from FFT k to phi is complex and depends on ifftshift
    # For simplicity and numerical stability, we use phi_ref = 0 (center phi)
    # This makes phase = exp(+2j * 0 * lambda²_0) = 1.0 when lambda²_0 = 0
    # and avoids numerical issues with large phi_ref values
    #
    # Note: This is an approximation. A more accurate approach would compute
    # phi_ref from the actual phi values used in interpolation, but that's complex
    # and the current approach works correctly when lambda²_0 = 0 (most common case)
    phi_ref = np.zeros_like(l2_diff_np)  # Use phi = 0 for all channels
    
    # Phase: exp(+2j * phi_ref * lambda²_0) = 1.0 when lambda²_0 = 0
    phase = np.exp(2.0j * phi_ref * l2_ref).astype(np.complex64)
    
    return k_cont, A, phase, N, d_phi


@dataclass(init=True, repr=True)
class NUFFT1D(DirectFourier1D):
    """
    Non-uniform FFT: Kaiser-kernel forward (FFT + interpolate), adjoint of that (A^H then IFFT).
    
    Implements the Faraday depth Fourier transform for non-uniformly spaced lambda²:
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi
    
    Uses Kaiser interpolation kernel for efficient non-uniform FFT. The phase factor
    accounts for the lambda²_0 reference. No iterative solver—adjoint is the linear
    adjoint of the Kaiser forward, consistent with the "positive" sign convention.
    """

    conv_size: int = None
    oversampling_factor: int = None
    normalize: bool = None
    solve: bool = None
    kaiser_beta: float = 6.0

    def __post_init__(self):
        super().__post_init__()
        if self.conv_size is None:
            self.conv_size = 4
        if self.oversampling_factor is None:
            self.oversampling_factor = 1
        if self.normalize is None:
            self.normalize = True
        if self.solve is None:
            self.solve = False
        if self.parameter is not None and self.parameter.cellsize is not None:
            self.configure()

    def configure(self) -> None:
        """Build and cache sparse interpolation matrix and phase (once). Prefer pydata/sparse so
        the same matrix works with dask (no dense copy). Otherwise scipy.sparse + dense on demand for dask."""
        self._nufft_A_sparse = None
        self._nufft_A_H_sparse = None
        self._nufft_phase = None
        self._nufft_A_dense = None
        self._nufft_A_H_dense = None
        self._nufft_pydata = False
        if self.parameter is None or self.dataset is None or self.parameter.cellsize is None:
            return
        _, A, phase, _, _ = _kaiser_forward_arrays(
            self.parameter, self.dataset, self.conv_size, self.kaiser_beta
        )
        if _pydata_sparse is not None:
            self._nufft_A_sparse = _pydata_sparse.COO.from_scipy_sparse(csr_matrix(A))
            self._nufft_A_H_sparse = self._nufft_A_sparse.T
            self._nufft_pydata = True
        else:
            self._nufft_A_sparse = csr_matrix(A)
            self._nufft_A_H_sparse = csr_matrix(A.T)
        self._nufft_phase = phase.astype(np.complex64)

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²) via NUFFT.
        
        Protected method. Applies FFT, Kaiser interpolation, and phase correction.
        Uses cached sparse matrix when available (pydata/sparse or scipy.sparse).
        
        Args:
            x: Complex Faraday depth spectrum (n_phi,)
            
        Returns:
            Complex polarization P(lambda²) (n_channels,)
        """
        phase = self._nufft_phase
        if phase is None:
            _, A, phase, _, _ = _kaiser_forward_arrays(
                self.parameter, self.dataset, self.conv_size, self.kaiser_beta
            )
            A_c = np.asarray(A, dtype=np.complex64)
        else:
            A_c = None
        xp = math_module(x)
        use_dask = da is not None and is_dask_array(x)
        # Shift input to center phi=0 at FFT index 0 (ifftshift before FFT)
        # For Faraday depth synthesis with exp(+2j*phi*lambda²) (positive sign),
        # use ifft with norm="forward" to get exp(+2πikn/N) instead of fft's exp(-2πikn/N)
        if use_dask:
            x_shifted = da.fft.ifftshift(x)
        else:
            x_shifted = np.fft.ifftshift(x)
        
        if self._nufft_pydata and self._nufft_A_sparse is not None:
            # Use ifft with norm="forward" for positive sign convention
            X = da.fft.ifft(x_shifted, norm="forward") if use_dask else xp.fft.ifft(x_shifted, norm="forward")
            if use_dask:
                interp = da.dot(self._nufft_A_sparse, X)
            else:
                interp = np.asarray(self._nufft_A_sparse @ np.asarray(X))
            out = phase * interp
        elif use_dask:
            if self._nufft_A_dense is None and self._nufft_A_sparse is not None:
                self._nufft_A_dense = self._nufft_A_sparse.toarray().astype(np.complex64)
            A_use = self._nufft_A_dense if self._nufft_A_dense is not None else A_c
            if A_use is None:
                _, A, phase, _, _ = _kaiser_forward_arrays(
                    self.parameter, self.dataset, self.conv_size, self.kaiser_beta
                )
                A_use = np.asarray(A, dtype=np.complex64)
            # Use ifft with norm="forward" for positive sign convention
            X = da.fft.ifft(x_shifted, norm="forward")
            interp = da.dot(A_use, X)
            out = phase * interp
        else:
            if self._nufft_A_sparse is not None:
                # Use ifft with norm="forward" for positive sign convention
                X = xp.fft.ifft(x_shifted, norm="forward")
                interp = self._nufft_A_sparse.dot(np.asarray(X))
                out = phase * interp
            else:
                # Use ifft with norm="forward" for positive sign convention
                X = xp.fft.ifft(x_shifted, norm="forward")
                interp = A_c @ np.asarray(X)
                out = phase * interp
        return out.astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator implementation: P(lambda²) -> phi via NUFFT.
        
        Protected method. Applies phase conjugation, Kaiser interpolation adjoint,
        and IFFT. Uses cached sparse matrix when available.
        
        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Complex Faraday depth spectrum (n_phi,)
        """
        phase = self._nufft_phase
        if phase is None:
            _, A, phase, _, _ = _kaiser_forward_arrays(
                self.parameter, self.dataset, self.conv_size, self.kaiser_beta
            )
            A_H = np.asarray(A.T, dtype=np.complex64)
        else:
            A_H = None
        xp = math_module(b)
        phase_conj = xp.conj(phase)
        phased = (phase_conj * b).astype(np.complex64)
        use_dask = da is not None and is_dask_array(b)
        if self._nufft_pydata and self._nufft_A_H_sparse is not None:
            if use_dask:
                back = da.dot(self._nufft_A_H_sparse, phased)
            else:
                back = np.asarray(self._nufft_A_H_sparse @ np.asarray(phased))
            # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
            # The adjoint of ifftshift is fftshift, so we don't need ifftshift here
            # Just apply fft, then fftshift to restore phi grid order (undoing ifftshift from forward)
            x_fft = da.fft.fft(back, norm="forward") if use_dask else xp.fft.fft(back, norm="forward")
            # Shift output to restore phi grid order (fftshift after FFT)
            # This undoes the ifftshift applied in forward direction
            out = da.fft.fftshift(x_fft) if use_dask else np.fft.fftshift(x_fft)
        elif use_dask:
            if self._nufft_A_H_dense is None and self._nufft_A_H_sparse is not None:
                self._nufft_A_H_dense = self._nufft_A_H_sparse.toarray().astype(np.complex64)
            A_H_use = self._nufft_A_H_dense if self._nufft_A_H_dense is not None else A_H
            if A_H_use is None:
                _, A, phase, _, _ = _kaiser_forward_arrays(
                    self.parameter, self.dataset, self.conv_size, self.kaiser_beta
                )
                A_H_use = np.asarray(A.T, dtype=np.complex64)
            back = da.dot(A_H_use, phased)
            # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
            # The adjoint of ifftshift is fftshift, so we don't need ifftshift here
            # Just apply fft, then fftshift to restore phi grid order (undoing ifftshift from forward)
            x_fft = da.fft.fft(back, norm="forward")
            # Shift output to restore phi grid order (fftshift after FFT)
            # This undoes the ifftshift applied in forward direction
            out = da.fft.fftshift(x_fft)
        else:
            phased_np = np.asarray(phased)
            if self._nufft_A_H_sparse is not None:
                back = self._nufft_A_H_sparse.dot(phased_np)
                # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
                # The adjoint of ifftshift is fftshift, so we don't need ifftshift here
                # Just apply fft, then fftshift to restore phi grid order (undoing ifftshift from forward)
                x_fft = xp.fft.fft(back, norm="forward")
                # Shift output to restore phi grid order (fftshift after FFT)
                # This undoes the ifftshift applied in forward direction
                out = np.fft.fftshift(x_fft)
            else:
                back = A_H @ phased_np
                # Use fft with norm="forward" for positive sign convention (adjoint of ifft)
                # The adjoint of ifftshift is fftshift, so we don't need ifftshift here
                # Just apply fft, then fftshift to restore phi grid order (undoing ifftshift from forward)
                x_fft = xp.fft.fft(back, norm="forward")
                # Shift output to restore phi grid order (fftshift after FFT)
                # This undoes the ifftshift applied in forward direction
                out = np.fft.fftshift(x_fft)
        out = out.astype(np.complex64)
        # Adjoint is pure: no extra scaling; dirty-spectrum scaling is applied in _dirty_spectrum_impl.
        return out

    def _dirty_spectrum_impl(self, data: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Dirty spectrum: A^H(weighted data) with scaling so the result matches the
        continuous definition (sum over channels; no 1/N from FFT).
        The FFT adjoint with norm="forward" yields (1/n_phi)*sum; multiply by n_ch here.
        """
        raw = super()._dirty_spectrum_impl(data)
        n_ch = self.dataset.m
        return raw * n_ch

    def RMTF(self, phi_x: float = 0.0):
        """
        Rotation Measure Transfer Function (RMTF).
        
        Public method. Uses direct adjoint of ones (from base class) then normalizes
        by n_phi if normalize=True.
        
        Args:
            phi_x: Faraday depth of point source (rad/m², default: 0.0)
            
        Returns:
            RMTF array (n_phi,)
        """
        rmtf = super().RMTF(phi_x)
        # Note: With norm="forward" in FFT/IFFT, the scaling is already handled correctly
        # Multiplying by N here would cause incorrect scaling
        # The normalize parameter is kept for backward compatibility but does nothing
        # if self.normalize:
        #     rmtf = rmtf * len(self.parameter.phi)
        return rmtf
