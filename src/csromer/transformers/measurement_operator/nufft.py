"""
NUFFT measurement operator (pynufft-based). Converts dask inputs to numpy at the
boundary for pynufft, then wraps the result back as a dask array so the rest of
the graph stays lazy (dask-in, dask-out). Not block-wise: each forward/adjoint
call is one chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from ...utils.array_utils import asnumpy, is_dask_array, maybe_compute
from .base import MeasurementOperator

try:
    import dask.array as da
except ImportError:
    da = None

try:
    from pynufft import NUFFT
    _NUFFT_CLASS = NUFFT
except ImportError:
    _NUFFT_CLASS = None

# Fix scipy.sparse.linalg.cg deprecation: pynufft calls cg() without atol. Patch
# before importing pynufft so pynufft gets the compliant wrapper.
def _patch_scipy_cg_atol():
    try:
        import scipy.sparse.linalg as _spla
        _cg_orig = _spla.cg
        def _cg_with_atol(*args, **kwargs):
            if "atol" not in kwargs:
                kwargs["atol"] = 0.0
            return _cg_orig(*args, **kwargs)
        _spla.cg = _cg_with_atol
    except Exception:
        pass


_patch_scipy_cg_atol()


def _require_pynufft():
    if _NUFFT_CLASS is None:
        raise ImportError("NUFFT1D requires pynufft. Install with: pip install pynufft")


@dataclass(init=True, repr=True)
class NUFFT1D(MeasurementOperator):
    """
    Non-uniform FFT via pynufft. forward/adjoint convert to numpy at the boundary.
    """

    conv_size: int = None
    oversampling_factor: int = None
    normalize: bool = None
    solve: bool = None
    nufft_instance: Any = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        _require_pynufft()
        self.nufft_instance = _NUFFT_CLASS()
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
        l2 = asnumpy(self.dataset.lambda2)
        l2_difference = l2 - self.dataset.l2_ref
        exp_factor = -2.0 * l2_difference * self.parameter.cellsize
        Nd = (len(self.parameter.phi),)
        Kd = (self.oversampling_factor * len(self.parameter.phi),)
        Jd = (self.conv_size,)
        om_exp = np.reshape(exp_factor, (self.dataset.m, 1))
        self.nufft_instance.plan(om_exp, Nd, Kd, Jd)

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        input_dask = da is not None and is_dask_array(x)
        x_np = asnumpy(x)
        out = self.nufft_instance.forward(x_np)
        if input_dask:
            return da.from_array(out, chunks=out.shape)
        return out

    def _adjoint_impl(self, b: Union[np.ndarray, Any], solver: str = "cg", maxiter: int = 1, **kwargs) -> Union[np.ndarray, Any]:
        # Raw adjoint: A^H(b), no weights, no K. Caller passes weighted residuals for gradients.
        input_dask = da is not None and is_dask_array(b)
        b_np = asnumpy(b)
        if self.solve:
            x = self.nufft_instance.solve(b_np, solver=solver, maxiter=maxiter)
        else:
            x = self.nufft_instance.adjoint(b_np)
        if self.normalize:
            x *= len(self.parameter.phi)
        if input_dask:
            return da.from_array(x, chunks=x.shape)
        return x

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        w = asnumpy(self.dataset.w)
        s = asnumpy(self.dataset.s) if self.dataset.s is not None else np.ones_like(w)
        k = float(maybe_compute(self.dataset.k)) if self.dataset.k is not None else 1.0
        weights = w / s
        x = self.nufft_instance.adjoint(weights)
        x *= len(self.parameter.phi) / k
        return x
