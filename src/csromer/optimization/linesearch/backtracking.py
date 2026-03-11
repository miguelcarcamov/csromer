"""Backtracking Armijo line search (Pyralysis-style). Used by CG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from ...utils.array_utils import maybe_compute
from .f1dim import f1dim
from .linesearcher import LineSearcher


@dataclass(init=True, repr=True)
class BacktrackingArmijo(LineSearcher):
    """Backtracking with Armijo rule: f(x + alpha*d) <= f(x) + c1*alpha*grad'*d."""

    contraction: float = 1e-4
    decrease: float = 0.5
    min_step: float = 1e-10

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        self._read_kwargs(**kwargs)
        current_phi = self.objective_function.phi
        f = f1dim(self.objective_function, x)
        grad = self.objective_function.dphi
        grad_norm = np.real(np.vdot(np.ravel(grad), np.ravel(grad)))
        grad_norm = float(maybe_compute(grad_norm))
        m = -self.contraction * grad_norm
        step_size = self._get_initial_step_size(x)
        for _ in range(self.max_iter):
            f_step = f(step_size)
            if hasattr(f_step, "compute"):
                f_step = float(f_step.compute())
            else:
                f_step = float(np.asarray(f_step).item())
            if f_step - current_phi > step_size * m:
                step_size *= self.decrease
                if step_size < self.min_step:
                    break
            else:
                break
        return f_step, step_size

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "contraction" in kwargs:
            self.contraction = kwargs["contraction"]
        if "decrease" in kwargs:
            self.decrease = kwargs["decrease"]
        if "min_step" in kwargs:
            self.min_step = kwargs["min_step"]
