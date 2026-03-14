"""
1D convolution utilities for real and complex arrays.

Used by Faraday depth restoration (convolution of model with clean beam).
Uses astropy.convolution.convolve_fft for FFT-based convolution (same-shape output).
"""
from __future__ import annotations

import numpy as np
from astropy.convolution import convolve_fft


def convolve(data: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Convolve a real 1D array with a real 1D kernel.

    Uses astropy.convolution.convolve_fft. Output shape equals input shape
    (equivalent to mode='same'). Kernel is not normalized here; pass
    peak- or sum-normalized kernel as needed.

    Args:
        data: 1D real array to convolve.
        kernel: 1D real kernel (e.g. Gaussian clean beam).
        mode: Ignored; kept for API compatibility. convolve_fft always returns same shape.

    Returns:
        Convolved 1D real array, same length as data.
    """
    data_np = np.asarray(data, dtype=np.float64)
    kernel_np = np.asarray(kernel, dtype=np.float64)
    if data_np.ndim != 1 or kernel_np.ndim != 1:
        raise ValueError("convolve expects 1D data and 1D kernel")
    out = convolve_fft(
        data_np,
        kernel_np,
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        normalize_kernel=False,
        crop=True,
    )
    out_dtype = getattr(np.asarray(data), "dtype", np.float32)
    if not np.issubdtype(out_dtype, np.floating):
        out_dtype = np.float32
    return np.asarray(out, dtype=out_dtype)


def convolve_complex(
    data: np.ndarray, kernel: np.ndarray, mode: str = "same"
) -> np.ndarray:
    """
    Convolve a complex 1D array with a real 1D kernel.

    Real and imaginary parts are convolved separately so that multiple
    peaks stay separated (e.g. for Faraday depth restoration). Uses
    astropy.convolution.convolve_fft for each part.

    Args:
        data: 1D complex array to convolve.
        kernel: 1D real kernel (e.g. Gaussian clean beam).
        mode: Ignored; kept for API compatibility. Output length equals len(data).

    Returns:
        Convolved 1D complex array, same length as data.
    """
    data_np = np.asarray(data, dtype=np.complex128)
    kernel_np = np.asarray(kernel, dtype=np.float64)
    if data_np.ndim != 1 or kernel_np.ndim != 1:
        raise ValueError("convolve_complex expects 1D data and 1D kernel")
    kwargs = dict(
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        normalize_kernel=False,
        crop=True,
    )
    real_conv = convolve_fft(data_np.real, kernel_np, **kwargs)
    imag_conv = convolve_fft(data_np.imag, kernel_np, **kwargs)
    out_dtype = getattr(np.asarray(data), "dtype", np.complex64)
    if not np.issubdtype(out_dtype, np.complexfloating):
        out_dtype = np.complex64
    return (real_conv + 1j * imag_conv).astype(out_dtype)
