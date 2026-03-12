"""
Simulation pipeline steps.

Each step runs with ctx = Faraday source (FaradaySource / FaradayThinSource etc.).
"""
from __future__ import annotations


class SimulateStep:
    """Call ctx.simulate() to generate polarization data."""

    def run(self, ctx) -> None:
        ctx.simulate()


class ApplyNoiseStep:
    """Call ctx.apply_noise(noise, random_state)."""

    def __init__(self, noise: float = None, random_state=None):
        self.noise = noise
        self.random_state = random_state

    def run(self, ctx) -> None:
        if self.noise is not None:
            ctx.apply_noise(noise=self.noise, random_state=self.random_state)


class ApplyRFIStep:
    """Remove a fraction of channels to simulate RFI. remove_frac = fraction of channels to remove (e.g. 0.1 = 10%)."""

    def __init__(self, remove_frac: float = 0.0, random_state=None, chunksize: int = None):
        self.remove_frac = remove_frac
        self.random_state = random_state
        self.chunksize = chunksize

    def run(self, ctx) -> None:
        if self.remove_frac > 0.0:
            # Source expects fraction to remove (0.1 = remove 10%, keep 90%)
            ctx.remove_channels(
                remove_frac=self.remove_frac,
                random_state=self.random_state,
                chunksize=self.chunksize,
            )
