"""Fixed step size line search (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from .f1dim import f1dim
from .linesearcher import LineSearcher


@dataclass(init=True, repr=True)
class Fixed(LineSearcher):
    """Returns a fixed step size."""

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        super()._read_kwargs(**kwargs)
        f = f1dim(self.objective_function, x)
        step_size = self._get_initial_step_size(x)
        fstep = f(step_size)
        if hasattr(fstep, "compute"):
            fstep = float(fstep.compute())
        else:
            fstep = float(np.asarray(fstep).item())
        return fstep, step_size
