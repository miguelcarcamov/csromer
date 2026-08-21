"""GLL (Grippo-Lampariello-Lucidi) Armijo: non-monotonic (Pyralysis-style)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ...reconstruction.parameter import Parameter
from .backtracking import BacktrackingArmijo
from .f1dim import f1dim


@dataclass(init=True, repr=True)
class GLLArmijo(BacktrackingArmijo):
    """GLL: f(x+alpha*d) <= max_{j in window} f(x_{k-j}) + c1*alpha*grad'*d."""

    memory_window: int = 5
    _function_history: np.ndarray = field(default=None, init=False, repr=False)
    _history_index: int = field(default=0, init=False, repr=False)
    _history_filled: bool = field(default=False, init=False, repr=False)
    _current_max: float = field(default=float("-inf"), init=False, repr=False)

    def _update_function_history(self, current_value: float) -> None:
        if self.memory_window == 0:
            self._function_history = None
            self._history_index = 0
            self._history_filled = False
            self._current_max = float("-inf")
            return
        if self._function_history is None:
            self._function_history = np.full(self.memory_window, np.nan, dtype=np.float32)
            self._history_index = 0
            self._history_filled = False
            self._current_max = float("-inf")
        old_value = self._function_history[self._history_index]
        self._function_history[self._history_index] = current_value
        if self._history_filled and not np.isnan(old_value):
            if old_value >= self._current_max:
                self._current_max = float(np.nanmax(self._function_history))
            else:
                self._current_max = max(self._current_max, current_value)
        else:
            self._current_max = max(self._current_max, current_value)
        self._history_index = (self._history_index + 1) % self.memory_window
        if self._history_index == 0:
            self._history_filled = True

    def _get_reference_value(self) -> float:
        if self._function_history is None or self._current_max == float("-inf"):
            return float("inf")
        return self._current_max

    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        self._read_kwargs(**kwargs)
        current_phi = self.objective_function.phi
        f = f1dim(self.objective_function, x)
        grad = self.objective_function.dphi
        grad_norm = np.real(np.vdot(np.ravel(grad), np.ravel(grad)))
        grad_norm = float(grad_norm.compute()) if hasattr(grad_norm,
                                                          "compute") else float(np.real(grad_norm))
        m = -self.contraction * grad_norm
        self._update_function_history(current_phi)
        step_size = self._get_initial_step_size(x)
        for _ in range(self.max_iter):
            f_step = f(step_size)
            if hasattr(f_step, "compute"):
                f_step = float(f_step.compute())
            else:
                f_step = float(np.asarray(f_step).item())
            ref = self._get_reference_value()
            if f_step - ref > step_size * m:
                step_size *= self.decrease
                if step_size < self.min_step:
                    break
            else:
                break
        return f_step, step_size

    def _read_kwargs(self, **kwargs) -> None:
        super()._read_kwargs(**kwargs)
        if "memory_window" in kwargs:
            self.memory_window = kwargs["memory_window"]

    def reset_history(self) -> None:
        self._function_history = None
        self._history_index = 0
        self._history_filled = False
        self._current_max = float("-inf")
