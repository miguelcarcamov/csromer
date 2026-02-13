"""
Gridded FFT for uniformly spaced lambda². Uses da.fft.fft / da.fft.ifft
when inputs are dask arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from ...utils.array_utils import is_dask_array
from .base import MeasurementOperator

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(init=True, repr=True)
class GriddedFFT1D(MeasurementOperator):
    """
    FFT-based measurement operator when lambda² is on a regular grid.
    forward(x) = FFT(x), adjoint(b) = IFFT(b). Uses da.fft.fft/ifft for dask arrays.
    """

    def configure(self) -> None:
        pass

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        if da is not None and is_dask_array(x):
            return da.fft.fft(x).astype(np.complex64)
        return np.fft.fft(x).astype(np.complex64)

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        if da is not None and is_dask_array(b):
            return da.fft.ifft(b).astype(np.complex64)
        return np.fft.ifft(b).astype(np.complex64)

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        n = self.parameter.phi.shape[0]
        if da is not None:
            return da.ones(n, dtype=np.complex64, chunks=(n,))
        return np.ones(n, dtype=np.complex64)
