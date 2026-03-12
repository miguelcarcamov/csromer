"""
Simulation pipeline.

Run an ordered sequence of steps on a Faraday source (ctx).
Use run_simulation(source, steps) or build steps from simulation.steps.
"""
from __future__ import annotations

from csromer.pipelines.core import run_pipeline


def run_simulation(ctx, steps: list) -> None:
    """Run the simulation pipeline: execute each step in order on ctx (Faraday source)."""
    run_pipeline(ctx, steps)
