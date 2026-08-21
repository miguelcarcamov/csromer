"""
1D function for line search: f(alpha) = F(x - alpha * dphi).
Used by CG and gradient-based line searchers. Convention: dphi is the gradient,
so we minimize along -dphi (descent).
"""
from __future__ import annotations

import numpy as np


def f1dim(objective_function, parameter):
    """
    Build 1D function f(alpha) = F(parameter.data - alpha * objective_function.dphi).

    Caller must set objective_function.dphi (e.g. via calculate_gradient) before calling.
    No projection/mask; works with numpy and dask arrays.
    """
    grad = objective_function.dphi
    data = parameter.data
    if grad is None:
        raise ValueError(
            "objective_function.dphi must be set before f1dim (e.g. call calculate_gradient)"
        )

    def wrapped(alpha: float) -> float:
        x = data - alpha * grad
        v = objective_function.evaluate(x)
        if hasattr(v, "compute"):
            return float(v.compute())
        return float(np.asarray(v).item())

    return wrapped
