"""
NUFFT measurement operator: Kaiser-kernel forward (FFT + interpolate), adjoint of forward (no iterative solver).
Forward uses FFT + Kaiser interpolation; adjoint is IFFT(A^H (phase* b)) so gradient is consistent.
No optimization algorithm in the adjoint—just the linear adjoint of the Kaiser forward.

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
    """Kaiser kernel value at offset u; zero for |u| > half_width."""
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
    """Build (n_ch, n_phi) matrix A so that A @ X interpolates X at continuous indices k_cont.
    Returns float32 to save memory; only ~(2*half_width+1) nonzeros per row."""
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
    """Compute k_cont, A, phase for forward/adjoint. Returns (k_cont, A, phase, N, d_phi)."""
    N = len(parameter.phi)
    d_phi = float(parameter.cellsize)
    l2_diff = dataset.lambda2 - dataset.l2_ref
    l2_diff_np = np.asarray(maybe_compute(l2_diff))
    n_ch = l2_diff_np.size
    k_cont = -N * d_phi * l2_diff_np / np.pi
    A = _build_kaiser_interp_matrix(n_ch, N, k_cont, conv_size, kaiser_beta)
    phase = np.exp(1j * np.pi * k_cont).astype(np.complex64)
    return k_cont, A, phase, N, d_phi


@dataclass(init=True, repr=True)
class NUFFT1D(DirectFourier1D):
    """
    Non-uniform FFT: Kaiser-kernel forward (FFT + interpolate), adjoint of that (A^H then IFFT).
    No iterative solver—adjoint is the linear adjoint of the Kaiser forward.
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
        if self._nufft_pydata and self._nufft_A_sparse is not None:
            X = da.fft.fft(x) if use_dask else xp.fft.fft(x)
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
            X = da.fft.fft(x)
            interp = da.dot(A_use, X)
            out = phase * interp
        else:
            if self._nufft_A_sparse is not None:
                X = xp.fft.fft(x)
                interp = self._nufft_A_sparse.dot(np.asarray(X))
                out = phase * interp
            else:
                X = xp.fft.fft(x)
                interp = A_c @ np.asarray(X)
                out = phase * interp
        return out.astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
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
            out = da.fft.ifft(back) if use_dask else xp.fft.ifft(back)
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
            out = da.fft.ifft(back)
        else:
            phased_np = np.asarray(phased)
            if self._nufft_A_H_sparse is not None:
                back = self._nufft_A_H_sparse.dot(phased_np)
                out = xp.fft.ifft(back)
            else:
                back = A_H @ phased_np
                out = xp.fft.ifft(back)
        out = out.astype(np.complex64)
        if self.normalize:
            out = out * len(self.parameter.phi)
        return out

    def RMTF(self, phi_x: float = 0.0):
        """RMTF: use direct adjoint of ones (same as base) then normalize."""
        rmtf = super().RMTF(phi_x)
        if self.normalize:
            rmtf = rmtf * len(self.parameter.phi)
        return rmtf
