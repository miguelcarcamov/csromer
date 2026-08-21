"""Golden section search (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from ...reconstruction.parameter import Parameter
from .linesearcher import LineSearcher
from .setup_bracketing import setup_bracketing

INV_GOLD = 1.0 / 1.618034


def pure_golden_section_search(
    f: Callable[[float], float],
    a: float,
    c: float,
    max_iter: int,
    tol: float,
    zeps: float = 1e-10,
) -> Tuple[float, float]:
    """Pure golden section search. Returns (f_min, x_min)."""
    x1 = c - INV_GOLD * (c - a)
    x2 = a + INV_GOLD * (c - a)
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(max_iter):
        if abs(c - a) < tol * (abs(x1) + abs(x2)) + zeps:
            return (f1, x1) if f1 < f2 else (f2, x2)
        if f2 < f1:
            a, x1, f1 = x1, x2, f2
            x2 = a + INV_GOLD * (c - a)
            f2 = f(x2)
        else:
            c, x2, f2 = x2, x1, f1
            x1 = c - INV_GOLD * (c - a)
            f1 = f(x1)
    return (f1, x1) if f1 < f2 else (f2, x2)


@dataclass(init=True, repr=True)
class GoldenSection(LineSearcher):
    """Golden section search (derivative-free)."""

    zeps: float = 1e-10

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        self._read_kwargs(**kwargs)
        f, a, c = setup_bracketing(self.objective_function, x, initial_a=0.0, initial_b=1.0)
        f_min, x_min = pure_golden_section_search(f, a, c, self.max_iter, self.tol, self.zeps)
        return f_min, x_min

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "zeps" in kwargs:
            self.zeps = kwargs["zeps"]
