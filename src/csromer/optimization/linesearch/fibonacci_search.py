"""Fibonacci search (Pyralysis-style, derivative-free)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from ...reconstruction.parameter import Parameter
from .linesearcher import LineSearcher
from .setup_bracketing import setup_bracketing


def _fib_sequence(n: int) -> list:
    if n <= 0:
        return [1]
    fib = [1, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


def pure_fibonacci_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 100,
    tol: float = 1e-7,
    zeps: float = 1e-10,
) -> Tuple[float, float]:
    """Fibonacci search. Returns (f_min, x_min)."""
    if b - a < tol:
        x_min = (a + b) / 2
        return f(x_min), x_min
    fib = _fib_sequence(max_iter)
    n = min(len(fib) - 1, max_iter)
    rho = fib[n - 1] / fib[n] if n > 0 else 0.5
    x1 = b - rho * (b - a)
    x2 = a + rho * (b - a)
    f1, f2 = f(x1), f(x2)
    for k in range(n - 1, 0, -1):
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            rho = fib[k - 1] / fib[k]
            x1 = b - rho * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            rho = fib[k - 1] / fib[k]
            x2 = a + rho * (b - a)
            f2 = f(x2)
        if abs(b - a) < tol * (abs(x1) + abs(x2)) + zeps:
            break
    if f1 < f2:
        return f1, x1
    return f2, x2


@dataclass(init=True, repr=True)
class Fibonacci(LineSearcher):
    """Fibonacci search (derivative-free)."""

    zeps: float = 1e-10

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        self._read_kwargs(**kwargs)
        f, a, c = setup_bracketing(self.objective_function, x, initial_a=0.0, initial_b=1.0)
        f_min, x_min = pure_fibonacci_search(f, a, c, self.max_iter, self.tol, self.zeps)
        return f_min, x_min

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "zeps" in kwargs:
            self.zeps = kwargs["zeps"]
