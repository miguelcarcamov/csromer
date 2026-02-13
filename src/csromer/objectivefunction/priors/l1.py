from dataclasses import dataclass

import numpy as np

from ..fi import Fi
from ...utils.array_utils import math_module


def approx_abs(x, epsilon, xp=None):
    """Magnitude: real or complex (Faraday depth treated as complex). xp = numpy or dask.array."""
    if xp is None:
        xp = np
    if xp is np and hasattr(x, "compute"):
        x = x  # keep dask; iscomplexobj may need compute - use dask version
    if xp is np and hasattr(x, "__array__") and not hasattr(x, "compute"):
        x = np.asarray(x)
    if xp is np and np.iscomplexobj(x):
        return xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + epsilon)
    if xp is np:
        return xp.sqrt(x * x + epsilon)
    # dask path: support complex (use np.issubdtype for dtype check)
    if np.issubdtype(x.dtype, np.complexfloating):
        return xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + epsilon)
    return xp.sqrt(x * x + epsilon)


@dataclass(init=True, repr=True)
class L1(Fi):
    is_differentiable: bool = False

    def __post_init__(self):
        super().__post_init__()

    def evaluate(self, x, epsilon=np.finfo(np.float32).tiny):
        xp = math_module(x)
        mag = approx_abs(x, epsilon, xp=xp)
        result = xp.sum(mag)
        self._func_value = result
        return result

    def calculate_gradient(self, x, epsilon=np.finfo(np.float32).tiny):
        xp = math_module(x)
        mag = approx_abs(x, epsilon, xp=xp)
        g = x / mag
        self._grad_value = g
        return g

    def calculate_prox(self, x, nu=0):
        """Soft-threshold on magnitude (real or complex). When nu > 0, threshold = self.reg * nu."""
        xp = math_module(x)
        thresh = self.reg if nu == 0 else self.reg * nu
        mag = xp.abs(x)
        eps = np.finfo(np.float32).tiny
        scale = xp.maximum(1.0 - thresh / (mag + eps), 0.0)
        return (x * scale).astype(x.dtype)
