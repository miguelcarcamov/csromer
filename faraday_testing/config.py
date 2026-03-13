"""
Configuration: SKA bands, source parameters, noise/RFI/depolarization, plotting.
"""

from __future__ import annotations

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "teal": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#666666",
    "accent": "#D55E00",
}

def setup_matplotlib():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["figure.figsize"] = (10, 8)


# ---------------------------------------------------------------------------
# Faraday depth (plotting default xlim when phi_xlim not set)
# ---------------------------------------------------------------------------

PHI_MAX = 1000.0   # rad/m²


# ---------------------------------------------------------------------------
# SKA bands (CLI key -> full name -> freq + short label)
# ---------------------------------------------------------------------------

def _ska_low_freq():
    return da.arange(50e6, 350e6, 3.9e3, dtype=np.float64)

def _ska_mid_b2_freq():
    return da.arange(950e6, 1760e6, 13.44e3, dtype=np.float64)

def _ska_mid_b5a_freq():
    return da.arange(4.6e9, 8.5e9, 13.44e3, dtype=np.float64)

def _ska_mid_b5b_freq():
    return da.arange(8.3e9, 15.4e9, 13.44e3, dtype=np.float64)


# Internal band id (used in code) -> config
SKA_BANDS = {
    "SKA-LOW":      {"freq": _ska_low_freq,   "short": "LOW"},
    "SKA-MID B2":   {"freq": _ska_mid_b2_freq, "short": "B2"},
    "SKA-MID B5a":  {"freq": _ska_mid_b5a_freq, "short": "B5a"},
    "SKA-MID B5b":  {"freq": _ska_mid_b5b_freq, "short": "B5b"},
}

# CLI band choices: low, mid-b2, mid-b5a, mid-b5b -> internal name
BAND_CLI_TO_INTERNAL = {
    "low": "SKA-LOW",
    "mid-b2": "SKA-MID B2",
    "mid-b5a": "SKA-MID B5a",
    "mid-b5b": "SKA-MID B5b",
}

BAND_CHOICES = list(BAND_CLI_TO_INTERNAL.keys())


def get_band_internal(cli_key: str) -> str:
    if cli_key not in BAND_CLI_TO_INTERNAL:
        raise ValueError(f"Unknown band {cli_key!r}. Choose from: {BAND_CHOICES}")
    return BAND_CLI_TO_INTERNAL[cli_key]


def get_bands_to_run(cli_bands: list[str]) -> list[str]:
    """Resolve CLI band list to internal band names (order preserved)."""
    if not cli_bands or "all" in cli_bands:
        return list(SKA_BANDS.keys())
    return [get_band_internal(b) for b in cli_bands]


# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------

THIN_PARAMS = {
    "phi_gal": 15.0,
    "s_nu": 1.0,
    "spectral_idx": -0.7,
    "dchi": 0.0,
}

THICK_PARAMS = {
    "phi_fg": 10.0,
    "phi_center": 20.0,
    "s_nu": 1.0,
    "spectral_idx": -0.7,
}

MIXED_CONFIG = [
    {"type": "thin",  "phi_gal": -400.0, "s_nu": 0.5, "spectral_idx": -0.7, "dchi": 0.0},
    {"type": "thick", "phi_fg": 50.0, "phi_center": 400.0, "s_nu": 0.5, "spectral_idx": -0.7},
]


# ---------------------------------------------------------------------------
# RFI and noise
# ---------------------------------------------------------------------------

RFI_REMOVE_FRAC_PER_BAND = {
    "SKA-LOW": 0.30,
    "SKA-MID B2": 0.20,
    "SKA-MID B5a": 0.10,
    "SKA-MID B5b": 0.10,
}

TARGET_SNR_WORST = 10
NOISE_BAND_FACTOR = {
    "SKA-LOW": 1.2,
    "SKA-MID B2": 1.0,
    "SKA-MID B5a": 0.9,
    "SKA-MID B5b": 0.9,
}

DEPOL_SIGMA_RM_THIN = 5.0   # rad/m²
DEPOL_SIGMA_RM_THICK = 3.0  # rad/m²


# ---------------------------------------------------------------------------
# Reconstructor default
# ---------------------------------------------------------------------------

RECONSTRUCTOR_DEFAULT = "cg"
