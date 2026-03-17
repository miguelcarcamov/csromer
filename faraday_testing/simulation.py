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
    THIN_PARAMS,
    get_mixed_config_for_band,
    get_thick_params_for_band,
)


def get_noise_sigma_jy(band_name: str, source_type: str) -> float:
    """Noise sigma (Jy) for worst-case scenario in this band and source type."""
    if source_type == "thin":
        ref = THIN_PARAMS["s_nu"]
    elif source_type == "thick":
        ref = get_thick_params_for_band(band_name)["s_nu"]
    elif source_type == "mixed":
        mixed_cfg = get_mixed_config_for_band(band_name)
        ref = mixed_cfg[0]["s_nu"] + mixed_cfg[1]["s_nu"]
    else:
        raise ValueError(f"Unknown source_type '{source_type}'")
    factor = NOISE_BAND_FACTOR.get(band_name, 1.0)
    return (ref / TARGET_SNR_WORST) * factor


def _simulate_thin_clean(nu, band_name: str):
    sigma = get_noise_sigma_jy(band_name, "thin")
    rng = np.random.RandomState(50)
    src = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(src, [SimulateStep(), ApplyNoiseStep(sigma, random_state=rng)])
    return src


def _simulate_thin_rfi(nu, band_name: str):
    rng_rfi = np.random.RandomState(42)
    sigma = get_noise_sigma_jy(band_name, "thin")
    rng_noise = np.random.RandomState(51)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]
    src = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(
        src,
        [
            SimulateStep(),
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_noise),
        ],
    )
    return src


def _simulate_thin_depol(nu, band_name: str):
    sigma = get_noise_sigma_jy(band_name, "thin")
    rng = np.random.RandomState(52)
    src = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(src, [SimulateStep()])
    src.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THIN)
    run_simulation(src, [ApplyNoiseStep(sigma, random_state=rng)])
    return src


def _run_thin_sources(nu, band_name: str) -> dict:
    return {
        "thin_clean": _simulate_thin_clean(nu, band_name),
        "thin_rfi": _simulate_thin_rfi(nu, band_name),
        "thin_depol": _simulate_thin_depol(nu, band_name),
    }


def _simulate_thick_clean(nu, band_name: str):
    sigma = get_noise_sigma_jy(band_name, "thick")
    rng = np.random.RandomState(60)
    thick_params = get_thick_params_for_band(band_name)
    src = FaradayThickSource(nu=nu, **thick_params)
    run_simulation(src, [SimulateStep(), ApplyNoiseStep(sigma, random_state=rng)])
    return src


def _simulate_thick_rfi(nu, band_name: str):
    rng_rfi = np.random.RandomState(43)
    sigma = get_noise_sigma_jy(band_name, "thick")
    rng_noise = np.random.RandomState(61)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]
    thick_params = get_thick_params_for_band(band_name)
    src = FaradayThickSource(nu=nu, **thick_params)
    run_simulation(
        src,
        [
            SimulateStep(),
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_noise),
        ],
    )
    return src


def _simulate_thick_depol(nu, band_name: str):
    sigma = get_noise_sigma_jy(band_name, "thick")
    rng = np.random.RandomState(62)
    thick_params = get_thick_params_for_band(band_name)
    src = FaradayThickSource(nu=nu, **thick_params)
    run_simulation(src, [SimulateStep()])
    src.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THICK)
    run_simulation(src, [ApplyNoiseStep(sigma, random_state=rng)])
    return src


def _run_thick_sources(nu, band_name: str) -> dict:
    return {
        "thick_clean": _simulate_thick_clean(nu, band_name),
        "thick_rfi": _simulate_thick_rfi(nu, band_name),
        "thick_depol": _simulate_thick_depol(nu, band_name),
    }


def _simulate_mixed_clean(nu, band_name: str):
    sigma = get_noise_sigma_jy(band_name, "mixed")
    rng = np.random.RandomState(70)
    mixed_cfg = get_mixed_config_for_band(band_name)
    cfg_thin = {k: v for k, v in mixed_cfg[0].items() if k != "type"}
    cfg_thick = {k: v for k, v in mixed_cfg[1].items() if k != "type"}
    thin_c = FaradayThinSource(nu=nu, **cfg_thin)
    run_simulation(thin_c, [SimulateStep()])
    thick_c = FaradayThickSource(nu=nu, **cfg_thick)
    run_simulation(thick_c, [SimulateStep()])
    mixed = thin_c + thick_c
    run_simulation(mixed, [ApplyNoiseStep(sigma, random_state=rng)])
    return mixed


def _simulate_mixed_rfi(nu, band_name: str):
    rng_rfi = np.random.RandomState(44)
    sigma = get_noise_sigma_jy(band_name, "mixed")
    rng_noise = np.random.RandomState(71)
    remove_frac = RFI_REMOVE_FRAC_PER_BAND[band_name]
    mixed_cfg = get_mixed_config_for_band(band_name)
    cfg_thin = {k: v for k, v in mixed_cfg[0].items() if k != "type"}
    cfg_thick = {k: v for k, v in mixed_cfg[1].items() if k != "type"}
    thin_r = FaradayThinSource(nu=nu, **cfg_thin)
    run_simulation(thin_r, [SimulateStep()])
    thick_r = FaradayThickSource(nu=nu, **cfg_thick)
    run_simulation(thick_r, [SimulateStep()])
    mixed = thin_r + thick_r
    run_simulation(
        mixed,
        [
            ApplyRFIStep(remove_frac=remove_frac, random_state=rng_rfi),
            ApplyNoiseStep(sigma, random_state=rng_noise),
        ],
    )
    return mixed


def _run_mixed_sources(nu, band_name: str) -> dict:
    return {
        "mixed_clean": _simulate_mixed_clean(nu, band_name),
        "mixed_rfi": _simulate_mixed_rfi(nu, band_name),
    }


_SIMULATE_ONE = {
    "thin_clean": _simulate_thin_clean,
    "thin_rfi": _simulate_thin_rfi,
    "thin_depol": _simulate_thin_depol,
    "thick_clean": _simulate_thick_clean,
    "thick_rfi": _simulate_thick_rfi,
    "thick_depol": _simulate_thick_depol,
    "mixed_clean": _simulate_mixed_clean,
    "mixed_rfi": _simulate_mixed_rfi,
}


def simulate_one_source(key: str, nu, band_name: str):
    """
    Simulate a single experiment (one source key) for the band.
    Returns None for thick/mixed keys when band is SKA-LOW.
    """
    if band_name == "SKA-LOW" and key not in ("thin_clean", "thin_rfi", "thin_depol"):
        return None
    fn = _SIMULATE_ONE.get(key)
    if fn is None:
        return None
    return fn(nu, band_name)


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
