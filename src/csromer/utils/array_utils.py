"""
Utilities for array handling that support both NumPy and Dask arrays.

Used across csromer for dask array compatibility. Functions detect array type
and handle numpy/dask appropriately to maintain lazy computation when possible.
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
    Convert array to NumPy.

    Public utility function. If input is a Dask array, computes it. Otherwise
    returns as NumPy array (copy if needed).

    Args:
        arr: Input array (numpy, dask, or None)

    Returns:
        NumPy array or None
    """
    if arr is None:
        return None
    if HAS_DASK and isinstance(arr, da.Array):
        return np.asarray(arr.compute())
    return np.asarray(arr)


def is_dask_array(arr) -> bool:
    """
    Check if array is a Dask array.

    Public utility function.

    Args:
        arr: Input array

    Returns:
        True if arr is a Dask array, False otherwise
    """
    if not HAS_DASK:
        return False
    return isinstance(arr, da.Array)


def length_of(arr) -> int:
    """
    Return length along first axis.

    Public utility function. Supports both NumPy and Dask arrays.

    Args:
        arr: Input array (or None)

    Returns:
        Length (int) or 0 if arr is None
    """
    if arr is None:
        return 0
    return arr.shape[0]


def math_module(arr):
    """
    Return the math/array module appropriate for the given array.

    Public utility function. Returns numpy or dask.array based on input type.
    Use for element-wise math (sqrt, cos, sin, etc.) so dask arrays stay lazy.

    Args:
        arr: Input array (or None)

    Returns:
        numpy or dask.array module
    """
    if arr is None:
        return np
    if HAS_DASK and isinstance(arr, da.Array):
        return da
    return np


def zeros_like(arr, **kwargs):
    """
    Return zeros with same shape/dtype as arr, in the same backend.

    Public utility function. Returns numpy.zeros_like or da.zeros_like based
    on input type. Use so gradient accumulation stays dask when input is dask.

    Args:
        arr: Input array (or None)
        **kwargs: Additional arguments passed to zeros_like

    Returns:
        Zeros array (same backend as arr) or None
    """
    if arr is None:
        return None
    if HAS_DASK and isinstance(arr, da.Array):
        return da.zeros_like(arr, **kwargs)
    return np.zeros_like(arr, **kwargs)
