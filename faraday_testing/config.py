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

# Global plotting font sizes for paper-ready readability.
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
AXIS_TITLE_FONT_SIZE = 14
FIGURE_TITLE_FONT_SIZE = 16
# Intrinsic (ground-truth) FD model overlay style.
INTRINSIC_MODEL_ALPHA = 0.45
INTRINSIC_MODEL_COLOR = COLORS["purple"]


def setup_matplotlib():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
    plt.rcParams["axes.titlesize"] = AXIS_TITLE_FONT_SIZE
    plt.rcParams["figure.titlesize"] = FIGURE_TITLE_FONT_SIZE


# ---------------------------------------------------------------------------
# Faraday depth (plotting xlim ±value rad/m² per band; default when not set)
# ---------------------------------------------------------------------------

PHI_MAX = 1000.0   # rad/m² (fallback when band not in PHI_XLIM_BY_BAND)

# FD plot x-axis half-width per band (wider for B5a/B5b so thick peaks are visible).
PHI_XLIM_BY_BAND = {
    "SKA-LOW": 100.0,
    "SKA-MID B1": 1000.0,
    "SKA-MID B2": 1000.0,
    "SKA-MID B5a": 10000.0,
    # Wider window so intrinsic components do not sit beneath the legend.
    "SKA-MID B5b": 80000.0,
}


# ---------------------------------------------------------------------------
# SKA bands (CLI key -> full name -> freq + short label)
# ---------------------------------------------------------------------------

def _ska_low_freq():
    return da.arange(50e6, 350e6, 3.9e3, dtype=np.float32)

def _ska_mid_b1_freq():
    return da.arange(350e6, 1050e6, 13.44e3, dtype=np.float32)

def _ska_mid_b2_freq():
    return da.arange(950e6, 1760e6, 13.44e3, dtype=np.float32)

def _ska_mid_b5a_freq():
    return da.arange(4.6e9, 8.5e9, 13.44e3 * 10.0, dtype=np.float32)

def _ska_mid_b5b_freq():
    return da.arange(8.3e9, 15.4e9, 13.44e3 * 10.0, dtype=np.float32)


# Internal band id (used in code) -> config
SKA_BANDS = {
    "SKA-LOW":      {"freq": _ska_low_freq,   "short": "LOW"},
    "SKA-MID B1":   {"freq": _ska_mid_b1_freq, "short": "B1"},
    "SKA-MID B2":   {"freq": _ska_mid_b2_freq, "short": "B2"},
    "SKA-MID B5a":  {"freq": _ska_mid_b5a_freq, "short": "B5a"},
    "SKA-MID B5b":  {"freq": _ska_mid_b5b_freq, "short": "B5b"},
}

# CLI band choices: low, mid-b1, mid-b2, mid-b5a, mid-b5b -> internal name
BAND_CLI_TO_INTERNAL = {
    "low": "SKA-LOW",
    "mid-b1": "SKA-MID B1",
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

# Nominal Faraday depth resolution Δφ_nom [rad/m²] per band (RMTF FWHM).
DELTA_PHI_NOM_BY_BAND = {
    "SKA-LOW": 9.837e-02,
    "SKA-MID B1": 5.312,
    "SKA-MID B2": 4.909e01,
    "SKA-MID B5a": 1.153e03,
    "SKA-MID B5b": 3.742e03,
}

# Max scale [rad/m²] per band: maximum recoverable Faraday structure width.
# Thick source full width 2*phi_fg must be ≤ this. From band λ² coverage.
MAX_SCALE_BY_BAND = {
    "SKA-LOW": 4.282e00,
    "SKA-MID B1": 3.854e01,
    "SKA-MID B2": 1.083e02,
    "SKA-MID B5a": 2.525e03,
    "SKA-MID B5b": 8.290e03,
}

# Thick source: top-hat in FD with half-width phi_fg (full width 2*phi_fg).
# phi_fg = min(THICK_PHI_FG_SCALE * Δφ_nom, MAX_SCALE/2) so 2*phi_fg ≤ MAX_SCALE.
# Standalone: phi_center = 2*phi_fg (thick sits in positive phi). Mixed overrides phi_center.
THICK_PHI_FG_SCALE = 2.0
THICK_PHI_CENTER_MULT = 2.0


def get_thick_params_for_band(band_name: str) -> dict:
    """
    Thick source params: half-width phi_fg (full width 2*phi_fg ≤ max scale), center 2*phi_fg.
    Used as-is for standalone thick; mixed overrides phi_center to half_sep.
    """
    delta_phi = DELTA_PHI_NOM_BY_BAND.get(band_name)
    if delta_phi is None:
        delta_phi = 5.0
    phi_fg = float(THICK_PHI_FG_SCALE * delta_phi)
    max_scale = MAX_SCALE_BY_BAND.get(band_name)
    if max_scale is not None and max_scale > 0 and 2.0 * phi_fg > max_scale:
        phi_fg = float(0.5 * max_scale)
    phi_center = float(THICK_PHI_CENTER_MULT * phi_fg)
    return {
        "phi_fg": phi_fg,
        "phi_center": phi_center,
        "s_nu": 1.0,
        "spectral_idx": -0.7,
    }

# Mixed source: thin + thick. Fallback only when band not in DELTA_PHI_NOM_BY_BAND.
MIXED_CONFIG = [
    {"type": "thin",  "phi_gal": -400.0, "s_nu": 0.5, "spectral_idx": -0.7, "dchi": 0.0},
    {"type": "thick", "phi_fg": 50.0, "phi_center": 400.0, "s_nu": 0.5, "spectral_idx": -0.7},
]

# Mixed separation: thin at -half_sep, thick center at +half_sep.
# Gap between thin peak and thick's nearest edge = 2*half_sep - phi_fg. Require gap >= min_gap
# so the two are clearly recognizable (not one on top of the other).
MIXED_MIN_GAP_BEAMS = 3.0  # minimum gap in resolution elements (Δφ_nom) between thin and thick
MIXED_SEPARATION_IN_BEAMS = 4.0
MIXED_SEPARATION_BEAMS_BY_BAND = {
    "SKA-MID B5a": 8.0,
    "SKA-MID B5b": 8.0,
}


def get_mixed_config_for_band(band_name: str) -> list:
    """
    Mixed source: thin (delta) at -half_sep, thick (top-hat half-width phi_fg) at +half_sep.
    half_sep is chosen so (1) thick does not overlap thin, (2) gap >= MIXED_MIN_GAP_BEAMS * Δφ_nom
    so the two components are clearly recognizable.
    """
    delta_phi = DELTA_PHI_NOM_BY_BAND.get(band_name)
    if delta_phi is None:
        return MIXED_CONFIG
    n_beams = MIXED_SEPARATION_BEAMS_BY_BAND.get(band_name, MIXED_SEPARATION_IN_BEAMS)
    thick_params = get_thick_params_for_band(band_name)
    # Keep mixed components balanced in polarized peak: thin 0.5 + thick 0.5.
    thick_params["s_nu"] = 0.5
    phi_fg = thick_params["phi_fg"]
    # Gap = 2*half_sep - phi_fg. Need gap >= min_gap_beams * delta_phi and half_sep >= phi_fg.
    min_gap = MIXED_MIN_GAP_BEAMS * delta_phi
    half_sep_from_gap = (phi_fg + min_gap) / 2.0
    half_sep_from_beams = 0.5 * n_beams * delta_phi
    half_sep = max(half_sep_from_beams, half_sep_from_gap, phi_fg)
    return [
        {"type": "thin", "phi_gal": -half_sep, "s_nu": 0.5, "spectral_idx": -0.7, "dchi": 0.0},
        {"type": "thick", **{**thick_params, "phi_center": half_sep}},
    ]


# ---------------------------------------------------------------------------
# RFI and noise
# ---------------------------------------------------------------------------

RFI_REMOVE_FRAC_PER_BAND = {
    "SKA-LOW": 0.30,
    "SKA-MID B1": 0.20,
    "SKA-MID B2": 0.20,
    "SKA-MID B5a": 0.10,
    "SKA-MID B5b": 0.10,
}

TARGET_SNR_WORST = 10
NOISE_BAND_FACTOR = {
    "SKA-LOW": 1.2,
    "SKA-MID B1": 1.0,
    "SKA-MID B2": 1.0,
    "SKA-MID B5a": 0.9,
    "SKA-MID B5b": 0.9,
}

DEPOL_SIGMA_RM_THIN = 5.0   # rad/m²
DEPOL_SIGMA_RM_THICK = 3.0  # rad/m²


# ---------------------------------------------------------------------------
# Reconstructor default and choices
# ---------------------------------------------------------------------------

RECONSTRUCTOR_CHOICES = ("csromer", "cg", "clean")
RECONSTRUCTOR_DEFAULT = "csromer"
