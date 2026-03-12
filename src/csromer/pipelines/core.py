"""
Shared pipeline abstraction for csromer.

Used by both the simulation pipeline and the reconstruction pipeline.
A pipeline runs an ordered sequence of steps; each step receives a context
and mutates it (e.g. context = Faraday source for simulation, reconstructor for reconstruction).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Step(Protocol):
    """A single pipeline step."""

    def run(self, ctx: object) -> None:
        """Execute the step. Mutates ctx as needed."""
        ...


def run_pipeline(ctx: object, steps: list) -> None:
    """Run each step in order. Steps receive ctx."""
    for step in steps:
        step.run(ctx)
