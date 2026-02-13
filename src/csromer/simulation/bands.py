"""
SKAO and LOFAR frequency band configurations. Supports numpy and dask arrays
for large channel counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(frozen=True)
class BandConfig:
    """Frequency band configuration: range, channel width, optional channel count."""

    name: str
    freq_min_hz: float
    freq_max_hz: float
    channel_width_hz: Optional[float] = None
    n_channels: Optional[int] = None

    def __post_init__(self):
        if self.channel_width_hz is None and self.n_channels is None:
            raise ValueError("Provide either channel_width_hz or n_channels")

    def freq_array(self, use_dask: bool = False, chunks: Optional[Union[int, tuple]] = None):
        """
        Return frequency array in Hz (linear spacing).
        use_dask: if True and dask available, return dask array.
        chunks: chunk size for dask (default auto).
        """
        if self.n_channels is not None:
            n = self.n_channels
        else:
            n = int(round((self.freq_max_hz - self.freq_min_hz) / self.channel_width_hz)) + 1
        freqs = np.linspace(self.freq_min_hz, self.freq_max_hz, n)
        if use_dask and da is not None:
            return da.from_array(freqs, chunks=chunks or "auto")
        return freqs


# SKA-LOW: 50–350 MHz
SKA_LOW = BandConfig(
    name="SKA-LOW",
    freq_min_hz=50e6,
    freq_max_hz=350e6,
    channel_width_hz=1e6,
)

# SKA-MID B2: 950–1760 MHz
SKA_MID_B2 = BandConfig(
    name="SKA-MID-B2",
    freq_min_hz=950e6,
    freq_max_hz=1760e6,
    channel_width_hz=1e6,
)

# SKA-MID B5a: 4.6–8.5 GHz
SKA_MID_B5a = BandConfig(
    name="SKA-MID-B5a",
    freq_min_hz=4.6e9,
    freq_max_hz=8.5e9,
    channel_width_hz=1e6,
)

# SKA-MID B5b: 8.3–15.4 GHz
SKA_MID_B5b = BandConfig(
    name="SKA-MID-B5b",
    freq_min_hz=8.3e9,
    freq_max_hz=15.4e9,
    channel_width_hz=1e6,
)

# LOFAR Low Band: 10–90 MHz
LOFAR_LOW = BandConfig(
    name="LOFAR-Low",
    freq_min_hz=10e6,
    freq_max_hz=90e6,
    channel_width_hz=195312.5,
)

# LOFAR High Band: 110–250 MHz
LOFAR_HIGH = BandConfig(
    name="LOFAR-High",
    freq_min_hz=110e6,
    freq_max_hz=250e6,
    channel_width_hz=195312.5,
)

# Predefined list for iteration
ALL_BANDS = [SKA_LOW, SKA_MID_B2, SKA_MID_B5a, SKA_MID_B5b, LOFAR_LOW, LOFAR_HIGH]


def get_band(name: str) -> Optional[BandConfig]:
    """Return band config by name (case-insensitive)."""
    for b in ALL_BANDS:
        if b.name.upper() == name.upper():
            return b
    return None
