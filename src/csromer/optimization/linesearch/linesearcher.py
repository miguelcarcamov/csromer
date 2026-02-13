"""
Base class for line searchers (Pyralysis-style).
Used by CG and FISTA. Convention: search_direction = -dphi (dphi = gradient).
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from ...reconstruction.parameter import Parameter

if TYPE_CHECKING:
    from .seeders.base import StepSizeSeeder


@dataclass(init=True, repr=True)
class LineSearcher(metaclass=ABCMeta):
    """
    Base for line search methods.
    objective_function.dphi holds the gradient (search along -dphi).
    """

    objective_function = None
    step: Optional[float] = 1.0
    tol: float = 1.0e-7
    max_iter: int = 100
    seeder: Optional["StepSizeSeeder"] = None

    @abstractmethod
    def search(self, x: Parameter, **kwargs) -> Tuple[float, float]:
        """Return (function_value, step_size)."""
        raise NotImplementedError

    def _get_initial_step_size(self, x: Parameter) -> float:
        if self.seeder is None:
            return self.step or 1.0
        return self.seeder.estimate_step_size(x, self.objective_function)

    def _read_kwargs(self, **kwargs) -> None:
        if "step" in kwargs:
            self.step = kwargs["step"]
        if "tol" in kwargs:
            self.tol = kwargs["tol"]
        if "max_iter" in kwargs:
            self.max_iter = kwargs["max_iter"]
        if "seeder" in kwargs:
            self.seeder = kwargs["seeder"]
