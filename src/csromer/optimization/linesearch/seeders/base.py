"""Base seeder for initial step size (Pyralysis-style)."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from ....reconstruction.parameter import Parameter


@dataclass(init=True, repr=True)
class StepSizeSeeder(metaclass=ABCMeta):
    """Base class for step size estimation."""

    min_step: float = 1e-10
    init_step: float = 1.0

    @abstractmethod
    def estimate_step_size(self, x: Parameter, objective_function) -> float:
        """Estimate initial step size for line search."""
        raise NotImplementedError
