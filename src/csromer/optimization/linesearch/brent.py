"""Brent's method for line search (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from .linesearcher import LineSearcher
from .setup_bracketing import setup_bracketing

CGOLD = (1.0 / 1.618034) ** 2


def _golden_step(x: float, a: float, b: float, xm: float) -> Tuple[float, float]:
    e = a - x if x >= xm else b - x
    d = CGOLD * e
    return e, d


@dataclass(init=True, repr=True)
class Brent(LineSearcher):
    """Brent's method (golden section + parabolic interpolation)."""

    zeps: float = 1e-10

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        self._read_kwargs(**kwargs)
        f, a, b, b_ = setup_bracketing(
            self.objective_function, x,
            initial_a=0.0, initial_b=1.0, return_middle=True
        )
        e = d = 0.0
        x_pt = w = v = b_
        fx = fw = fv = f(x_pt)
        for _ in range(self.max_iter):
            xm = 0.5 * (a + b)
            tol1 = self.tol * abs(x_pt) + self.zeps
            tol2 = 2.0 * tol1
            if abs(x_pt - xm) <= (tol2 - 0.5 * (b - a)):
                return fx, x_pt
            if abs(e) > tol1:
                r = (x_pt - w) * (fx - fv)
                q = (x_pt - v) * (fx - fw)
                p = (x_pt - v) * q - (x_pt - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = abs(q)
                e_old, e = e, d
                if abs(p) >= abs(0.5 * q * e_old) or p <= q * (a - x_pt) or p >= q * (b - x_pt):
                    e, d = _golden_step(x_pt, a, b, xm)
                else:
                    d = p / q
                    u = x_pt + d
                    if (u - a) < tol2 or (b - u) < tol2:
                        d = np.copysign(tol1, xm - x_pt)
            else:
                e, d = _golden_step(x_pt, a, b, xm)
            u = x_pt + d if abs(d) >= tol1 else x_pt + np.copysign(tol1, d)
            fu = f(u)
            if fu <= fx:
                if u >= x_pt:
                    a = x_pt
                else:
                    b = x_pt
                v, w, x_pt = w, x_pt, u
                fv, fw, fx = fw, fx, fu
            else:
                if u < x_pt:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x_pt:
                    v, w = w, u
                    fv, fw = fw, fu
                elif fu <= fv or v == x_pt or v == w:
                    v = u
                    fv = fu
        return fx, x_pt

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "zeps" in kwargs:
            self.zeps = kwargs["zeps"]
