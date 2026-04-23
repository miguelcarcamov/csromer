"""
Shared 1D FFT helpers used across measurement and restoration code.
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from .array_utils import is_dask_array

try:
    import dask.array as da
except ImportError:
    da = None


def fft1d_forward(
    x: Union[np.ndarray, Any],
    *,
    centered: bool = False,
    norm: str = "forward",
) -> Union[np.ndarray, Any]:
    """
    1D forward transform wrapper with optional centred-grid ordering.

    Uses ``ifft`` as the forward transform to match Burn-sign conventions used in
    this codebase. When ``centered=True``, applies ``ifftshift`` first so arrays
    stored with phi=0 at the centre index map correctly to FFT order.
    """
    if da is not None and is_dask_array(x):
        x_in = da.fft.ifftshift(x) if centered else x
        return da.fft.ifft(x_in, norm=norm).astype(np.complex64)
    x_in = np.fft.ifftshift(x) if centered else x
    return np.fft.ifft(x_in, norm=norm).astype(np.complex64)


def fft1d_inverse(
    x: Union[np.ndarray, Any],
    *,
    centered: bool = False,
    norm: str = "forward",
) -> Union[np.ndarray, Any]:
    """
    1D inverse transform wrapper paired with :func:`fft1d_forward`.

    Uses ``fft`` as the inverse pair for Burn-sign conventions. When
    ``centered=True``, applies ``fftshift`` after transform to return arrays in
    centred phi-grid ordering.
    """
    if da is not None and is_dask_array(x):
        out = da.fft.fft(x, norm=norm).astype(np.complex64)
        return da.fft.fftshift(out).astype(np.complex64) if centered else out
    out = np.fft.fft(x, norm=norm).astype(np.complex64)
    return np.fft.fftshift(out).astype(np.complex64) if centered else out
