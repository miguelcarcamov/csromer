"""
Utilities for array handling that support both NumPy and Dask arrays.
Used across csromer for dask array compatibility.
"""
from __future__ import annotations

from typing import Union

import numpy as np

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def asnumpy(arr) -> np.ndarray:
    """
    Convert array to NumPy. If it is a Dask array, compute it.
    Otherwise return as NumPy array (copy if needed).
    """
    if arr is None:
        return None
    if HAS_DASK and isinstance(arr, da.Array):
        return np.asarray(arr.compute())
    return np.asarray(arr)


def is_dask_array(arr) -> bool:
    """Return True if arr is a Dask array."""
    if not HAS_DASK:
        return False
    return isinstance(arr, da.Array)


def maybe_compute(scalar_or_array):
    """
    If the argument is a Dask array (including 0-d), return the computed value.
    Otherwise return as-is.
    """
    if scalar_or_array is None:
        return None
    if HAS_DASK and isinstance(scalar_or_array, da.Array):
        return scalar_or_array.compute()
    return scalar_or_array


def length_of(arr) -> int:
    """Return length along first axis, supporting both NumPy and Dask arrays."""
    if arr is None:
        return 0
    return arr.shape[0]


def math_module(arr):
    """
    Return the math/array module (numpy or dask.array) appropriate for the given array.
    Use for element-wise math (sqrt, cos, sin, sinc, etc.) so dask arrays stay lazy.
    """
    if arr is None:
        return np
    if HAS_DASK and isinstance(arr, da.Array):
        return da
    return np


def zeros_like(arr, **kwargs):
    """
    Return zeros with same shape/dtype as arr, in the same backend (numpy or dask).
    Use so gradient accumulation stays dask when input is dask.
    """
    if arr is None:
        return None
    if HAS_DASK and isinstance(arr, da.Array):
        return da.zeros_like(arr, **kwargs)
    return np.zeros_like(arr, **kwargs)
