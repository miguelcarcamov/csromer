"""
1D convolution utilities for real and complex arrays.

Used by Faraday depth restoration (convolution of model with clean beam).
Uses the shared 1D FFT procedure from the gridded measurement operator.
"""
from __future__ import annotations

import numpy as np

from .fft_utils import fft1d_forward, fft1d_inverse


def _same_slice(n_full: int, n_data: int) -> slice:
    """
    Slice to recover mode='same' output from full linear convolution.

    Matches ``np.convolve(data, kernel, mode="same")`` alignment for 1D arrays.
    """
    start = (n_full - n_data) // 2
    return slice(start, start + n_data)


def convolve(data: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Convolve a real 1D array with a real 1D kernel.

    Uses zero-padded FFT-domain multiplication (linear convolution). Output shape
    equals input shape (equivalent to mode='same'). Kernel is not normalized here;
    pass peak- or sum-normalized kernel as needed.

    Args:
        data: 1D real array to convolve.
        kernel: 1D real kernel (e.g. Gaussian clean beam).
        mode: Only "same" is supported.

    Returns:
        Convolved 1D real array, same length as data.
    """
    data_np = np.asarray(data, dtype=np.float64)
    kernel_np = np.asarray(kernel, dtype=np.float64)
    if data_np.ndim != 1 or kernel_np.ndim != 1:
        raise ValueError("convolve expects 1D data and 1D kernel")
    if mode != "same":
        raise ValueError("convolve only supports mode='same'")
    n_data = data_np.shape[0]
    n_kernel = kernel_np.shape[0]
    n_full = n_data + n_kernel - 1
    spec_data = fft1d_forward(np.pad(data_np, (0, n_full - n_data)), centered=False, norm="forward")
    spec_kernel = fft1d_forward(
        np.pad(kernel_np, (0, n_full - n_kernel)), centered=False, norm="forward"
    )
    full = fft1d_inverse(spec_data * spec_kernel, centered=False, norm="forward")
    out = np.asarray(full[_same_slice(n_full, n_data)].real, dtype=np.float64)
    out_dtype = getattr(np.asarray(data), "dtype", np.float32)
    if not np.issubdtype(out_dtype, np.floating):
        out_dtype = np.float32
    return np.asarray(out, dtype=out_dtype)


def convolve_complex(data: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Convolve a complex 1D array with a real 1D kernel.

    Real and imaginary parts are convolved separately so that multiple
    peaks stay separated (e.g. for Faraday depth restoration). Uses
    shared FFT-domain convolution for each part.

    Args:
        data: 1D complex array to convolve.
        kernel: 1D real kernel (e.g. Gaussian clean beam).
        mode: Only "same" is supported.

    Returns:
        Convolved 1D complex array, same length as data.
    """
    data_np = np.asarray(data, dtype=np.complex128)
    kernel_np = np.asarray(kernel, dtype=np.float64)
    if data_np.ndim != 1 or kernel_np.ndim != 1:
        raise ValueError("convolve_complex expects 1D data and 1D kernel")
    if mode != "same":
        raise ValueError("convolve_complex only supports mode='same'")
    n_data = data_np.shape[0]
    n_kernel = kernel_np.shape[0]
    n_full = n_data + n_kernel - 1
    kernel_spec = fft1d_forward(
        np.pad(kernel_np, (0, n_full - n_kernel)), centered=False, norm="forward"
    )
    data_real_spec = fft1d_forward(
        np.pad(data_np.real, (0, n_full - n_data)), centered=False, norm="forward"
    )
    data_imag_spec = fft1d_forward(
        np.pad(data_np.imag, (0, n_full - n_data)), centered=False, norm="forward"
    )
    real_full = fft1d_inverse(data_real_spec * kernel_spec, centered=False, norm="forward")
    imag_full = fft1d_inverse(data_imag_spec * kernel_spec, centered=False, norm="forward")
    data_slice = _same_slice(n_full, n_data)
    real_conv = np.asarray(real_full[data_slice].real, dtype=np.float64)
    imag_conv = np.asarray(imag_full[data_slice].real, dtype=np.float64)
    out_dtype = getattr(np.asarray(data), "dtype", np.complex64)
    if not np.issubdtype(out_dtype, np.complexfloating):
        out_dtype = np.complex64
    return (real_conv + 1j * imag_conv).astype(out_dtype)
