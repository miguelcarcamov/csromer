"""
Base measurement operator: maps between Faraday depth (model) space and
lambda² (data) space. Subclasses implement _forward_impl and _adjoint_impl.

When wavelet_transform is set, forward expects PyWT coefficients: coefficients
→ (pywt reconstruct) → Faraday depth → (measurement) → P(lambda²). Adjoint then
returns coefficients (adjoint of measurement → decompose).
"""
from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ...utils.array_utils import math_module, maybe_compute

if TYPE_CHECKING:
    from ...base import Dataset
    from ...dictionaries import Wavelet
    from ...reconstruction import Parameter


@dataclass(init=True, repr=True)
class MeasurementOperator(metaclass=ABCMeta):
    """
    Base class for 1D measurement operators (Faraday depth <-> lambda²).
    When wavelet_transform is set, forward(x) expects coefficients; adjoint(b) returns coefficients.
    """
    dataset: "Dataset" = None
    parameter: "Parameter" = None
    wavelet_transform: Optional["Wavelet"] = None

    def __post_init__(self):
        if self.parameter is not None:
            self.parameter = copy.deepcopy(self.parameter)

    @abstractmethod
    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Map from complex Faraday depth (model) to data (lambda²). Subclass implementation."""
        pass

    @abstractmethod
    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """Map from data (lambda²) to complex Faraday depth. Subclass implementation."""
        pass

    def forward(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Raw forward: F (Faraday depth) -> P(lambda²), no weights, no K."""
        if self.wavelet_transform is not None:
            x = self.wavelet_transform.reconstruct_complex(np.array(x, copy=False))
        return self._forward_impl(x)

    def adjoint(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """Map from P(lambda²) to Faraday depth (or coefficients if wavelet_transform set)."""
        res = self._adjoint_impl(b, **kwargs)
        if self.wavelet_transform is not None:
            # Complex Faraday depth -> complex coefficients (pywt)
            res = self.wavelet_transform.decompose_complex(res)
        return res

    def backward(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """Raw adjoint: backward(b) = A^H b. Used in gradients with weighted residuals."""
        return self.adjoint(b, **kwargs)

    def dirty_spectrum(self, data: Union[np.ndarray, Any] = None) -> Union[np.ndarray, Any]:
        """Dirty Faraday depth spectrum: A^H(weighted data) / K. Use for initial dirty map only."""
        if data is None and self.dataset is not None:
            data = self.dataset.data
        return self._dirty_spectrum_impl(data)

    def _dirty_spectrum_impl(self, data: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Dirty map: A^H(weighted data) / K. Uses dataset w, s, k when available.
        Only K is forced to float; w, s, data stay lazy (dask) so the graph does not grow."""
        if self.dataset is None:
            return self.adjoint(data)
        w = self.dataset.w
        s = getattr(self.dataset, "s", None)
        xp = math_module(w)
        s = s if s is not None else xp.ones_like(w)
        k = getattr(self.dataset, "k", None)
        k = float(maybe_compute(k)) if k is not None else 1.0
        weighted = (w / s) * data
        return self.adjoint(weighted) / k

    def configure(self) -> None:
        """Optional: configure operator (e.g. NUFFT plan). Override if needed."""
        pass

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        """Optional: rotation measure transfer function. Override in subclass if supported."""
        raise NotImplementedError("RMTF not implemented for this operator")
