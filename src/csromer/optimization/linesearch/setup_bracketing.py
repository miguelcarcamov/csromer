"""Setup bracketing for Brent and Golden Section (f1dim + mnbrak)."""
from __future__ import annotations

from typing import Callable, Tuple

from ...reconstruction.parameter import Parameter
from .f1dim import f1dim
from .bracketing import mnbrak


def setup_bracketing(
    objective_function,
    x: Parameter,
    initial_a: float = 0.0,
    initial_b: float = 1.0,
    return_middle: bool = False,
) -> Tuple[Callable[[float], float], float, float, ...]:
    """Bracket a minimum: f, a, c (and optionally b_)."""
    f = f1dim(objective_function, x)
    a_, b_, c_, _, _, _ = mnbrak(f, initial_a, initial_b)
    a = min(a_, c_)
    c = max(a_, c_)
    if return_middle:
        return f, a, c, b_
    return f, a, c
