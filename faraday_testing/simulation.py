"""
Simulate thin/thick/mixed sources with clean, RFI, and depolarization variants.
"""

from __future__ import annotations

import numpy as np

from csromer.pipelines import (
    ApplyNoiseStep,
    ApplyRFIStep,
    SimulateStep,
    run_simulation,
)
from csromer.simulation import FaradayThickSource, FaradayThinSource

from faraday_testing.config import (
    DEPOL_SIGMA_RM_THIN,
    DEPOL_SIGMA_RM_THICK,
    MIXED_CONFIG,
    NOISE_BAND_FACTOR,
    RFI_REMOVE_FRAC_PER_BAND,
    TARGET_SNR_WORST,
    THICK_PARAMS,
    THIN_PARAMS,
)


def get_noise_sigma_jy(band_name: str, source_type: str) -> float:
    """Noise sigma (Jy) for worst-case scenario in this band and source type."""
    if source_type == "thin":
        ref = THIN_PARAMS["s_nu"]
    elif source_type == "thick":
        ref = THICK_PARAMS["s_nu"]
    elif source_type == "mixed":
        ref = MIXED_CONFIG[0]["s_nu"] + MIXED_CONFIG[1]["s_nu"]
    else:
        raise ValueError(f"Unknown source_type '{source_type}'")
    factor = NOISE_BAND_FACTOR.get(band_name, 1.0)
    return (ref / TARGET_SNR_WORST) * factor


def _run_thin_sources(nu, band_name: str) -> dict:
    rng_rfi = np.random.RandomState(42)
    sigma = get_noise_sigma_jy(band_name, "thin")
    rng_clean = np.random.RandomState(50)
    rng_rfi_noise = np.random.RandomState(51)
    rng_depol = np.random.RandomState(52)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]

    thin_clean = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(thin_clean, [SimulateStep(), ApplyNoiseStep(sigma, random_state=rng_clean)])

    thin_rfi = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(
        thin_rfi,
        [
            SimulateStep(),
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_rfi_noise),
        ],
    )

    thin_depol = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(thin_depol, [SimulateStep()])
    thin_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THIN)
    run_simulation(thin_depol, [ApplyNoiseStep(sigma, random_state=rng_depol)])

    return {"thin_clean": thin_clean, "thin_rfi": thin_rfi, "thin_depol": thin_depol}


def _run_thick_sources(nu, band_name: str) -> dict:
    rng_rfi = np.random.RandomState(43)
    sigma = get_noise_sigma_jy(band_name, "thick")
    rng_clean = np.random.RandomState(60)
    rng_rfi_noise = np.random.RandomState(61)
    rng_depol = np.random.RandomState(62)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]

    thick_clean = FaradayThickSource(nu=nu, **THICK_PARAMS)
    run_simulation(thick_clean, [SimulateStep(), ApplyNoiseStep(sigma, random_state=rng_clean)])

    thick_rfi = FaradayThickSource(nu=nu, **THICK_PARAMS)
    run_simulation(
        thick_rfi,
        [
            SimulateStep(),
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_rfi_noise),
        ],
    )

    thick_depol = FaradayThickSource(nu=nu, **THICK_PARAMS)
    run_simulation(thick_depol, [SimulateStep()])
    thick_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THICK)
    run_simulation(thick_depol, [ApplyNoiseStep(sigma, random_state=rng_depol)])

    return {"thick_clean": thick_clean, "thick_rfi": thick_rfi, "thick_depol": thick_depol}


def _run_mixed_sources(nu, band_name: str) -> dict:
    rng_rfi = np.random.RandomState(44)
    sigma = get_noise_sigma_jy(band_name, "mixed")
    rng_clean = np.random.RandomState(70)
    rng_rfi_noise = np.random.RandomState(71)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]
    cfg_thin = {k: v for k, v in MIXED_CONFIG[0].items() if k != "type"}
    cfg_thick = {k: v for k, v in MIXED_CONFIG[1].items() if k != "type"}

    thin_c = FaradayThinSource(nu=nu, **cfg_thin)
    run_simulation(thin_c, [SimulateStep()])
    thick_c = FaradayThickSource(nu=nu, **cfg_thick)
    run_simulation(thick_c, [SimulateStep()])
    mixed_clean = thin_c + thick_c
    run_simulation(mixed_clean, [ApplyNoiseStep(sigma, random_state=rng_clean)])

    thin_r = FaradayThinSource(nu=nu, **cfg_thin)
    run_simulation(thin_r, [SimulateStep()])
    thick_r = FaradayThickSource(nu=nu, **cfg_thick)
    run_simulation(thick_r, [SimulateStep()])
    mixed_rfi = thin_r + thick_r
    run_simulation(
        mixed_rfi,
        [
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_rfi_noise),
        ],
    )

    return {"mixed_clean": mixed_clean, "mixed_rfi": mixed_rfi}


def simulate_sources_for_band(nu, band_name: str) -> dict:
    """
    Simulate thin/thick/mixed, clean/RFI/depol for a band.
    For SKA-LOW only thin sources are produced; thick/mixed are None.
    """
    out = _run_thin_sources(nu, band_name)
    if band_name == "SKA-LOW":
        out["thick_clean"] = out["thick_rfi"] = out["thick_depol"] = None
        out["mixed_clean"] = out["mixed_rfi"] = None
    else:
        out.update(_run_thick_sources(nu, band_name))
        out.update(_run_mixed_sources(nu, band_name))
    return out
