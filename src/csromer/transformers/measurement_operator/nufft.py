"""
NUFFT measurement operator: Kaiser-kernel forward (FFT + interpolate), adjoint of forward.

Implements the Faraday depth Fourier transform (Burn 1966):
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

No l2_ref in forward/adjoint. Adjoint is linear adjoint of the Kaiser forward.

Efficiency: interpolation matrix A is built once in configure() and stored as sparse. When
pydata/sparse is installed, A is stored as sparse.COO so the same matrix works for both numpy
and dask (da.dot(sparse, dask_vector) via dask's tensordot_lookup). Otherwise we use scipy.sparse
for numpy and build dense A on demand for dask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import i0

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
        # Normalize row so sum_j A[c,j] = 1; then A^H(weighted) sums to sum(weighted)
        # and dirty spectrum amplitude matches DirectFourier/GriddedFFT.
        row_sum = float(np.sum(A[c, :]))
        if row_sum > 1e-15:
            A[c, :] /= row_sum
    return A


def _kaiser_forward_arrays(parameter, dataset, conv_size: int, kaiser_beta: float):
    """
    Compute k_cont, A, phase for forward/adjoint.

    Private helper. No l2_ref in transform; uses lambda² directly for k_cont and phase=1.
    """
    N = len(parameter.phi)
    d_phi = float(parameter.cellsize)
    l2_np = np.asarray(maybe_compute(dataset.lambda2))
    n_ch = l2_np.size

    # Map lambda² to FFT k indices: k = N * d_phi * lambda² / π (no l2_ref)
    k_cont = N * d_phi * l2_np / np.pi
    A = _build_kaiser_interp_matrix(n_ch, N, k_cont, conv_size, kaiser_beta)
    # No l2_ref: phase = 1
    phase = np.ones(n_ch, dtype=np.complex64)
    return k_cont, A, phase, N, d_phi


@dataclass(init=True, repr=True)
class NUFFT1D(DirectFourier1D):
    """
    Non-uniform FFT: Kaiser-kernel forward (FFT + interpolate), adjoint of that (A^H then IFFT).

    Implements the Faraday depth Fourier transform (Burn 1966) for non-uniform lambda².
    No l2_ref in forward/adjoint.
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
        The FFT adjoint with norm="forward" yields (1/N)*sum; multiply by N (n_phi)
        so dirty amplitude matches DirectFourier/GriddedFFT (GriddedFFT uses n_chan = N).
        """
        raw = super()._dirty_spectrum_impl(data)
        n_phi = len(self.parameter.phi)
        return raw * n_phi

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
