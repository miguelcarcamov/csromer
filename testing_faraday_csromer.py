"""
CS-ROMER Faraday revision script: SKA bands, thin/thick/mixed sources,
RFI and depolarization, with 2×2 comparison figures matching testing_faraday.py:

- Clean vs RFI:    top row = normal (pol vs λ², FD spectrum), bottom row = RFI.
- Clean vs Depolarization: top row = normal, bottom row = depolarized.

Uses csromer pipelines (simulation + reconstruction); FD spectrum from reconstruction (dirty + restored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as C_LIGHT

from csromer.pipelines import (
    ApplyNoiseStep,
    ApplyRFIStep,
    CSROMERReconstructorWrapper,
    SimulateStep,
    make_cg_optimizer,
    make_fista_optimizer,
    run_simulation,
)
from csromer.pipelines.reconstruction.reconstruction_stats import calculate_fd_signal_noise
from csromer.simulation import FaradayThickSource, FaradayThinSource

# Speed of light in m/s (float)
c = float(C_LIGHT.value)


# Colorblind-friendly palette based on Paul Tol / ggplot colors.
# These are widely used in publications and remain distinguishable for
# common forms of color vision deficiency on both screen and paper.
COLORS = {
    # Core series colors
    "blue": "#0072B2",     # ggplot/Tol blue
    "orange": "#E69F00",   # Tol orange
    "teal": "#009E73",     # Tol green/teal
    "purple": "#CC79A7",   # Tol magenta
    "yellow": "#F0E442",   # Tol yellow (best for fills, not thin lines),

    # Neutrals and accents
    "black": "#000000",
    "gray": "#666666",
    "accent": "#D55E00",   # Tol red, used for peak / emphasis
}


# Matplotlib defaults
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.figsize"] = (10, 8)


def _peak_legend_label(peak_phi: float, peak_error: float | None) -> str:
    """Format peak φ with optional error for legend (rad/m²)."""
    if peak_error is not None and np.isfinite(peak_error):
        # If the error is comfortably representable with two decimal places,
        # show it in fixed-point; otherwise fall back to scientific notation.
        if abs(peak_error) >= 1e-2:
            err_str = f"{peak_error:.2f}"
        else:
            err_str = f"{peak_error:.2e}"
        return rf"Peak $\phi$ = {peak_phi:.2f} $\pm$ {err_str}"
    return rf"Peak $\phi$ = {peak_phi:.2f}"


def _sigma_from_amplitude(amp: np.ndarray, method: str = "median") -> float:
    """
    Estimate Gaussian (Ricean) σ from amplitude |F|.

    Both estimators give the same underlying σ (Re/Im noise level):
    - "median": σ = median(|F|) / sqrt(2 ln 2). Robust to outliers/structure.
    - "rms":    σ = RMS(|F|) / sqrt(2). Tracks full spread (noise + structure).
    """
    amp = np.asarray(amp)
    if method == "median":
        val = float(np.median(amp))
        if val <= 0.0 or not np.isfinite(val):
            return 0.0
        return val / np.sqrt(2.0 * np.log(2.0))
    elif method == "rms":
        rms = float(np.sqrt(np.mean(amp**2)))
        if not np.isfinite(rms) or rms <= 0:
            return 0.0
        return rms / np.sqrt(2.0)
    else:
        raise ValueError(f"_sigma_from_amplitude: method must be 'median' or 'rms', got {method!r}")


# SKA-like frequency bands (values copied from testing_faraday.py)
# Using numpy here; csromer simulation classes accept nu arrays.
def ska_low_freq():
    # 50–350 MHz, 3.9 kHz step
    return da.arange(50e6, 350e6, 3.9e3, dtype=np.float64)


def ska_mid_b2_freq():
    # 950–1760 MHz, 13.44 kHz step
    return da.arange(950e6, 1760e6, 13.44e3, dtype=np.float64)


def ska_mid_b5a_freq():
    # 4.6–8.5 GHz, 13.44 kHz step
    return da.arange(4.6e9, 8.5e9, 13.44e3, dtype=np.float64)


def ska_mid_b5b_freq():
    # 8.3–15.4 GHz, 13.44 kHz step
    return da.arange(8.3e9, 15.4e9, 13.44e3, dtype=np.float64)


SKA_BANDS = {
    "SKA-LOW": {"freq": ska_low_freq, "short": "LOW"},
    "SKA-MID B2": {"freq": ska_mid_b2_freq, "short": "B2"},
    "SKA-MID B5a": {"freq": ska_mid_b5a_freq, "short": "B5a"},
    "SKA-MID B5b": {"freq": ska_mid_b5b_freq, "short": "B5b"},
}


# Source / effect parameters (from testing_faraday main block)
THIN_PARAMS = {
    "phi_gal": 15.0,  # rad/m²
    # Use a more realistic polarized flux density (~10 mJy instead of 1 Jy).
    "s_nu": 1.0,
    "spectral_idx": -0.7,
    "dchi": 0.0,
}

THICK_PARAMS = {
    "phi_fg": 10.0,  # rad/m²
    "phi_center": 20.0,  # rad/m²
    # Match thin-source reference flux for consistency.
    "s_nu": 1.0,
    "spectral_idx": -0.7,
}

MIXED_CONFIG = [
    {
        "type": "thin",
        "phi_gal": -400.0,
        # Half of the thin/thick reference flux per component.
        "s_nu": 0.5,
        "spectral_idx": -0.7,
        "dchi": 0.0,
    },
    {
        "type": "thick",
        "phi_fg": 50.0,
        "phi_center": 400.0,
        "s_nu": 0.5,
        "spectral_idx": -0.7,
    },
]
RFI_REMOVE_FRAC = 0.10  # legacy default (kept for reference; not used directly below)

# Band-dependent RFI fractions (worst-case style scenarios).
# These represent relatively pessimistic RFI occupancies, to stress–test the reconstruction:
# - SKA-LOW: low-frequency, typically most RFI–affected  -> 30% of channels flagged
# - SKA-MID B2: moderate RFI environment                 -> 20% of channels flagged
# - SKA-MID B5a/B5b: higher frequency, generally cleaner -> 10% of channels flagged
RFI_REMOVE_FRAC_PER_BAND = {
    "SKA-LOW": 0.30,
    "SKA-MID B2": 0.20,
    "SKA-MID B5a": 0.10,
    "SKA-MID B5b": 0.10,
}

# Worst-case noise (sigma in Jy) per band and source type for FISTA/adaptive-λ.
# sigma = reference_intensity / target_SNR; band factor scales up noise for harder bands (LOW).
# Thin/thick: s_nu=1.0 Jy → base sigma at SNR 25; mixed: s_nu=0.5 per component → conservative.
TARGET_SNR_WORST = 10
NOISE_BAND_FACTOR = {"SKA-LOW": 1.2, "SKA-MID B2": 1.0, "SKA-MID B5a": 0.9, "SKA-MID B5b": 0.9}


def get_noise_sigma_jy(band_name: str, source_type: str) -> float:
    """
    Noise sigma (Jy) for worst-case scenario in this band and config (thin/thick/mixed).

    The reference intensity is now taken from the actual source amplitudes
    (THIN_PARAMS/THICK_PARAMS/MIXED_CONFIG) so that TARGET_SNR_WORST is
    interpreted consistently regardless of the chosen s_nu values.
    """
    if source_type == "thin":
        ref_intensity = THIN_PARAMS["s_nu"]
    elif source_type == "thick":
        ref_intensity = THICK_PARAMS["s_nu"]
    elif source_type == "mixed":
        # Effective reference flux for mixed case: sum of component fluxes.
        ref_intensity = MIXED_CONFIG[0]["s_nu"] + MIXED_CONFIG[1]["s_nu"]
    else:
        raise ValueError(f"Unknown source_type '{source_type}' in get_noise_sigma_jy")

    band_factor = NOISE_BAND_FACTOR.get(band_name, 1.0)
    return (ref_intensity / TARGET_SNR_WORST) * band_factor


DEPOL_SIGMA_RM_THIN = 5.0  # rad/m²
DEPOL_SIGMA_RM_THICK = 3.0  # rad/m²

# Reconstructor: "csromer" (FISTA + L1) or "cg" (conjugate gradient)
RECONSTRUCTOR = "csromer"

# Faraday grid parameters (same for all bands; tweak as needed)
PHI_MAX = 1000.0  # rad/m²
PHI_CELLSIZE = 0.5  # rad/m²


def simulate_sources_for_band(nu: np.ndarray, band_name: str):
    """Simulate thin/thick/mixed, clean/RFI/depol for a given band using pipeline steps."""
    rng_thin_rfi = np.random.RandomState(42)
    rng_thick_rfi = np.random.RandomState(43)
    rng_mixed_rfi = np.random.RandomState(44)
    sigma_thin = get_noise_sigma_jy(band_name, "thin")
    rng_thin_clean_noise = np.random.RandomState(50)
    rng_thin_rfi_noise = np.random.RandomState(51)
    rng_thin_depol_noise = np.random.RandomState(52)

    # Thin clean
    thin_clean = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(
        thin_clean,
        [SimulateStep(), ApplyNoiseStep(sigma_thin, random_state=rng_thin_clean_noise)],
    )

    # Thin RFI
    thin_rfi = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(
        thin_rfi,
        [
            SimulateStep(),
            ApplyRFIStep(
                remove_frac=RFI_REMOVE_FRAC_PER_BAND[band_name],
                random_state=rng_thin_rfi,
            ),
            ApplyNoiseStep(sigma_thin, random_state=rng_thin_rfi_noise),
        ],
    )

    # Thin depolarized
    thin_depol = FaradayThinSource(nu=nu, **THIN_PARAMS)
    run_simulation(thin_depol, [SimulateStep()])
    thin_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THIN)
    run_simulation(
        thin_depol,
        [ApplyNoiseStep(sigma_thin, random_state=rng_thin_depol_noise)],
    )

    thick_clean = thick_rfi = thick_depol = None
    mixed_clean = mixed_rfi = None

    # For SKA-LOW we only use thin sources (as in testing_faraday)
    if band_name != "SKA-LOW":
        sigma_thick = get_noise_sigma_jy(band_name, "thick")
        rng_thick_clean_noise = np.random.RandomState(60)
        rng_thick_rfi_noise = np.random.RandomState(61)
        rng_thick_depol_noise = np.random.RandomState(62)
        sigma_mixed = get_noise_sigma_jy(band_name, "mixed")
        rng_mixed_clean_noise = np.random.RandomState(70)
        rng_mixed_rfi_noise = np.random.RandomState(71)

        # Thick clean
        thick_clean = FaradayThickSource(nu=nu, **THICK_PARAMS)
        run_simulation(
            thick_clean,
            [SimulateStep(), ApplyNoiseStep(sigma_thick, random_state=rng_thick_clean_noise)],
        )

        # Thick RFI
        thick_rfi = FaradayThickSource(nu=nu, **THICK_PARAMS)
        run_simulation(
            thick_rfi,
            [
                SimulateStep(),
                ApplyRFIStep(
                    remove_frac=RFI_REMOVE_FRAC_PER_BAND[band_name],
                    random_state=rng_thick_rfi,
                ),
                ApplyNoiseStep(sigma_thick, random_state=rng_thick_rfi_noise),
            ],
        )

        # Thick depolarized
        thick_depol = FaradayThickSource(nu=nu, **THICK_PARAMS)
        run_simulation(thick_depol, [SimulateStep()])
        thick_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THICK)
        run_simulation(
            thick_depol,
            [ApplyNoiseStep(sigma_thick, random_state=rng_thick_depol_noise)],
        )

        # Mixed clean
        cfg_thin = {k: v for k, v in MIXED_CONFIG[0].items() if k != "type"}
        cfg_thick = {k: v for k, v in MIXED_CONFIG[1].items() if k != "type"}

        mixed_clean_thin = FaradayThinSource(nu=nu, **cfg_thin)
        run_simulation(mixed_clean_thin, [SimulateStep()])
        mixed_clean_thick = FaradayThickSource(nu=nu, **cfg_thick)
        run_simulation(mixed_clean_thick, [SimulateStep()])
        mixed_clean = mixed_clean_thin + mixed_clean_thick
        run_simulation(
            mixed_clean,
            [ApplyNoiseStep(sigma_mixed, random_state=rng_mixed_clean_noise)],
        )

        # Mixed RFI
        mixed_rfi_thin = FaradayThinSource(nu=nu, **cfg_thin)
        run_simulation(mixed_rfi_thin, [SimulateStep()])
        mixed_rfi_thick = FaradayThickSource(nu=nu, **cfg_thick)
        run_simulation(mixed_rfi_thick, [SimulateStep()])
        mixed_rfi = mixed_rfi_thin + mixed_rfi_thick
        run_simulation(
            mixed_rfi,
            [
                ApplyRFIStep(
                    remove_frac=RFI_REMOVE_FRAC_PER_BAND[band_name],
                    random_state=rng_mixed_rfi,
                ),
                ApplyNoiseStep(sigma_mixed, random_state=rng_mixed_rfi_noise),
            ],
        )

    return {
        "thin_clean": thin_clean,
        "thin_rfi": thin_rfi,
        "thin_depol": thin_depol,
        "thick_clean": thick_clean,
        "thick_rfi": thick_rfi,
        "thick_depol": thick_depol,
        "mixed_clean": mixed_clean,
        "mixed_rfi": mixed_rfi,
    }


def run_csromer_reconstruction(
    source,
    oversampling: float = 4.0,
    maxiter: int = 100 ,
    reconstructor: str = "csromer",
):
    """Run reconstruction on a single csromer Dataset using the pipeline reconstructor.

    Args:
        source: Dataset (e.g. simulated source).
        oversampling: Oversampling factor for Faraday depth grid.
        maxiter: Maximum iterations (FISTA or CG depending on reconstructor).
        reconstructor: "csromer" (FISTA + L1) or "cg".

    Returns:
        CSROMERReconstructorWrapper instance after reconstruct().
    """
    if reconstructor.lower() == "cg":
        optimizer_factory = make_cg_optimizer(
            maxiter=maxiter,
            tol=1e-12,
            verbose=True,
        )
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=1e-5,
            optimizer_factory=optimizer_factory,
        )
    else:
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=0.5,  # starting λ
            adaptive_lambda=True,
            target_chi2=1.0,
            lambda_update_gamma=0.5,
            max_lambda_updates=20,
            optimizer_factory=make_fista_optimizer(
                maxiter=maxiter,
                tol=1e-12,
                verbose=True,
                monotonic=True
            ),
        )
    recon.reconstruct()
    return recon


def plot_2x2_clean_vs_rfi(
    clean_source,
    rfi_source,
    recon_clean,
    recon_rfi,
    band_label: str,
    source_type: str,
    filename: str | None = None,
    phi_xlim: float | tuple[float, float] | None = None,
    figsize=(18, 12),
):
    """
    2×2 comparison: Clean (top) vs RFI (bottom).
    - (1) Top-left: Clean polarization vs λ² (irregular data)
    - (2) Top-right: double panel — top: FD spectrum (|dirty|, |restored|, 5σ, red peak); bottom: residuals
    - (3) Bottom-left: RFI polarization vs λ² (irregular data)
    - (4) Bottom-right: double panel — top: FD spectrum; bottom: residuals
    """
    import matplotlib.gridspec as gridspec

    def _xlim(phi_xlim):
        if phi_xlim is None:
            return (-PHI_MAX, PHI_MAX)
        if isinstance(phi_xlim, (tuple, list)) and len(phi_xlim) == 2:
            return (float(phi_xlim[0]), float(phi_xlim[1]))
        x = float(phi_xlim)
        return (-x, x)

    xlim_phi = _xlim(phi_xlim)
    l2_clean = np.asarray(clean_source.lambda2)
    data_clean = np.asarray(clean_source.data)
    l2_rfi = np.asarray(rfi_source.lambda2)
    data_rfi = np.asarray(rfi_source.data)

    phi_clean = np.asarray(recon_clean.parameter.phi)
    fd_dirty_clean = np.asarray(recon_clean.fd_dirty)
    fd_clean = np.asarray(recon_clean.fd_restored)
    fd_res_clean = np.asarray(recon_clean.fd_residual)
    phi_rfi = np.asarray(recon_rfi.parameter.phi)
    fd_dirty_rfi = np.asarray(recon_rfi.fd_dirty)
    fd_rfi = np.asarray(recon_rfi.fd_restored)
    fd_res_rfi = np.asarray(recon_rfi.fd_residual)

    # FD-spectrum noise: estimate from |F_dirty| assuming Rayleigh statistics,
    # so the 5σ line reflects the Ricean / Rayleigh amplitude noise level.
    sigma_clean = _sigma_from_amplitude(np.abs(fd_dirty_clean), method="median")
    sigma_rfi = _sigma_from_amplitude(np.abs(fd_dirty_rfi), method="median")
    sigma_res_clean = _sigma_from_amplitude(np.abs(fd_res_clean), method="rms")
    sigma_res_rfi = _sigma_from_amplitude(np.abs(fd_res_rfi), method="rms")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    # Right column: each cell is two stacked subplots (FD top, residuals bottom)
    gs_right_top = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)
    gs_right_bot = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)

    # (1) Top-left: Clean polarization vs λ²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(l2_clean, np.abs(data_clean), ".", color=COLORS["blue"], markersize=0.6, alpha=0.9, label=r"$|P|$")
    ax1.plot(l2_clean, data_clean.real, ".", color=COLORS["purple"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax1.plot(l2_clean, data_clean.imag, ".", color=COLORS["orange"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax1.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax1.set_ylabel("Polarization [Jy]", fontsize=11)
    ax1.set_title(r"Reference (no RFI): Polarization vs $\lambda^2$", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) Top-right: double — FD (abs only, red peak transparent) + residuals
    ax2_fd = fig.add_subplot(gs_right_top[0])
    ax2_fd.plot(phi_clean, np.abs(fd_dirty_clean), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax2_fd.plot(phi_clean, np.abs(fd_clean), "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax2_fd.axhline(5.0 * sigma_clean, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    peak_idx_c = np.argmax(np.abs(fd_clean))
    peak_phi_c = float(phi_clean[peak_idx_c])
    peak_err_c = getattr(recon_clean, "rm_restored_error", None)
    ax2_fd.axvline(peak_phi_c, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45, label=_peak_legend_label(peak_phi_c, peak_err_c))
    ax2_fd.set_xlim(xlim_phi[0], xlim_phi[1])
    ax2_fd.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax2_fd.set_title("Reference (no RFI): Faraday depth spectrum", fontsize=12, fontweight="bold")
    ax2_fd.legend(loc="best", fontsize=9)
    ax2_fd.grid(True, alpha=0.3)
    ax2_fd.tick_params(axis="x", labelbottom=False)
    ax2_res = fig.add_subplot(gs_right_top[1])
    ax2_res.plot(phi_clean, np.abs(fd_res_clean), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax2_res.plot(phi_clean, fd_res_clean.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax2_res.plot(phi_clean, fd_res_clean.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax2_res.axhline(sig * sigma_res_clean, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax2_res.axhline(-sig * sigma_res_clean, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax2_res.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax2_res.axvline(peak_phi_c, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax2_res.set_xlim(xlim_phi[0], xlim_phi[1])
    ax2_res.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax2_res.set_ylabel("Residuals", fontsize=11)
    ax2_res.grid(True, alpha=0.3)

    # (3) Bottom-left: RFI polarization vs λ²
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(l2_rfi, np.abs(data_rfi), ".", color=COLORS["blue"], markersize=0.6, alpha=0.9, label=r"$|P|$")
    ax3.plot(l2_rfi, data_rfi.real, ".", color=COLORS["purple"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax3.plot(l2_rfi, data_rfi.imag, ".", color=COLORS["orange"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax3.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax3.set_ylabel("Polarization [Jy]", fontsize=11)
    ax3.set_title(r"With RFI: Polarization vs $\lambda^2$", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (4) Bottom-right: double — FD (abs only) + residuals
    ax4_fd = fig.add_subplot(gs_right_bot[0])
    ax4_fd.plot(phi_rfi, np.abs(fd_dirty_rfi), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax4_fd.plot(phi_rfi, np.abs(fd_rfi), "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax4_fd.axhline(5.0 * sigma_rfi, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    peak_idx_r = np.argmax(np.abs(fd_rfi))
    peak_phi_r = float(phi_rfi[peak_idx_r])
    peak_err_r = getattr(recon_rfi, "rm_restored_error", None)
    ax4_fd.axvline(peak_phi_r, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45, label=_peak_legend_label(peak_phi_r, peak_err_r))
    ax4_fd.set_xlim(xlim_phi[0], xlim_phi[1])
    ax4_fd.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax4_fd.set_title("With RFI: Faraday depth spectrum", fontsize=12, fontweight="bold")
    ax4_fd.legend(loc="best", fontsize=9)
    ax4_fd.grid(True, alpha=0.3)
    ax4_fd.tick_params(axis="x", labelbottom=False)
    ax4_res = fig.add_subplot(gs_right_bot[1])
    ax4_res.plot(phi_rfi, np.abs(fd_res_rfi), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax4_res.plot(phi_rfi, fd_res_rfi.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax4_res.plot(phi_rfi, fd_res_rfi.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax4_res.axhline(sig * sigma_res_rfi, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax4_res.axhline(-sig * sigma_res_rfi, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax4_res.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax4_res.axvline(peak_phi_r, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax4_res.set_xlim(xlim_phi[0], xlim_phi[1])
    ax4_res.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax4_res.set_ylabel("Residuals", fontsize=11)
    ax4_res.grid(True, alpha=0.3)

    plt.suptitle(f"{source_type} Source: Reference vs RFI ({band_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_2x2_clean_vs_depol(
    clean_source,
    depol_source,
    recon_clean,
    recon_depol,
    band_label: str,
    source_type: str,
    filename: str | None = None,
    phi_xlim: float | tuple[float, float] | None = None,
    figsize=(18, 12),
):
    """
    2×2 comparison: Clean (top) vs Depolarized (bottom).
    - (1) Top-left: Clean polarization vs λ²
    - (2) Top-right: double — FD spectrum (|dirty|, |restored|, 5σ, red peak); bottom: residuals
    - (3) Bottom-left: Depolarized polarization vs λ²
    - (4) Bottom-right: double — FD spectrum; bottom: residuals
    """
    import matplotlib.gridspec as gridspec

    def _xlim(phi_xlim):
        if phi_xlim is None:
            return (-PHI_MAX, PHI_MAX)
        if isinstance(phi_xlim, (tuple, list)) and len(phi_xlim) == 2:
            return (float(phi_xlim[0]), float(phi_xlim[1]))
        return (-float(phi_xlim), float(phi_xlim))

    xlim_phi = _xlim(phi_xlim)
    l2_clean = np.asarray(clean_source.lambda2)
    data_clean = np.asarray(clean_source.data)
    l2_depol = np.asarray(depol_source.lambda2)
    data_depol = np.asarray(depol_source.data)

    phi_clean = np.asarray(recon_clean.parameter.phi)
    fd_dirty_clean = np.asarray(recon_clean.fd_dirty)
    fd_clean = np.asarray(recon_clean.fd_restored)
    fd_res_clean = np.asarray(recon_clean.fd_residual)
    phi_depol = np.asarray(recon_depol.parameter.phi)
    fd_dirty_depol = np.asarray(recon_depol.fd_dirty)
    fd_depol = np.asarray(recon_depol.fd_restored)
    fd_res_depol = np.asarray(recon_depol.fd_residual)

    # FD-spectrum noise from |F_dirty| assuming Rayleigh statistics (Ricean amplitude).
    sigma_clean = _sigma_from_amplitude(np.abs(fd_dirty_clean), method="median")
    sigma_depol = _sigma_from_amplitude(np.abs(fd_dirty_depol), method="median")
    sigma_res_clean = _sigma_from_amplitude(np.abs(fd_res_clean), method="rms")
    sigma_res_depol = _sigma_from_amplitude(np.abs(fd_res_depol), method="rms")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_right_top = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)
    gs_right_bot = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)

    # (1) Top-left: Clean polarization vs λ²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(l2_clean, np.abs(data_clean), ".", color=COLORS["blue"], markersize=0.6, alpha=0.9, label=r"$|P|$")
    ax1.plot(l2_clean, data_clean.real, ".", color=COLORS["blue"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax1.plot(l2_clean, data_clean.imag, ".", color=COLORS["blue"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax1.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax1.set_ylabel("Polarization [Jy]", fontsize=11)
    ax1.set_title(r"Reference (no depol.): Polarization vs $\lambda^2$", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) Top-right: double — FD (abs only, red peak transparent) + residuals
    ax2_fd = fig.add_subplot(gs_right_top[0])
    ax2_fd.plot(phi_clean, np.abs(fd_dirty_clean), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax2_fd.plot(phi_clean, np.abs(fd_clean), "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax2_fd.axhline(5.0 * sigma_clean, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    peak_idx_c = np.argmax(np.abs(fd_clean))
    peak_phi_c = float(phi_clean[peak_idx_c])
    peak_err_c = getattr(recon_clean, "rm_restored_error", None)
    ax2_fd.axvline(peak_phi_c, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45, label=_peak_legend_label(peak_phi_c, peak_err_c))
    ax2_fd.set_xlim(xlim_phi[0], xlim_phi[1])
    ax2_fd.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax2_fd.set_title("Reference (no depol.): Faraday depth spectrum", fontsize=12, fontweight="bold")
    ax2_fd.legend(loc="best", fontsize=9)
    ax2_fd.grid(True, alpha=0.3)
    ax2_fd.tick_params(axis="x", labelbottom=False)
    ax2_res = fig.add_subplot(gs_right_top[1])
    ax2_res.plot(phi_clean, np.abs(fd_res_clean), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax2_res.plot(phi_clean, fd_res_clean.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax2_res.plot(phi_clean, fd_res_clean.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax2_res.axhline(sig * sigma_res_clean, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax2_res.axhline(-sig * sigma_res_clean, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax2_res.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax2_res.axvline(peak_phi_c, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax2_res.set_xlim(xlim_phi[0], xlim_phi[1])
    ax2_res.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax2_res.set_ylabel("Residuals", fontsize=11)
    ax2_res.grid(True, alpha=0.3)

    # (3) Bottom-left: Depolarized polarization vs λ²
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(l2_depol, np.abs(data_depol), ".", color=COLORS["orange"], markersize=0.6, alpha=0.9, label=r"$|P|$")
    ax3.plot(l2_depol, data_depol.real, ".", color=COLORS["orange"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax3.plot(l2_depol, data_depol.imag, ".", color=COLORS["orange"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax3.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax3.set_ylabel("Polarization [Jy]", fontsize=11)
    ax3.set_title(r"Depolarized: Polarization vs $\lambda^2$", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (4) Bottom-right: double — FD (abs only) + residuals
    ax4_fd = fig.add_subplot(gs_right_bot[0])
    ax4_fd.plot(phi_depol, np.abs(fd_dirty_depol), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax4_fd.plot(phi_depol, np.abs(fd_depol), "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax4_fd.axhline(5.0 * sigma_depol, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    peak_idx_d = np.argmax(np.abs(fd_depol))
    peak_phi_d = float(phi_depol[peak_idx_d])
    peak_err_d = getattr(recon_depol, "rm_restored_error", None)
    ax4_fd.axvline(peak_phi_d, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45, label=_peak_legend_label(peak_phi_d, peak_err_d))
    ax4_fd.set_xlim(xlim_phi[0], xlim_phi[1])
    ax4_fd.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax4_fd.set_title("Depolarized: Faraday depth spectrum", fontsize=12, fontweight="bold")
    ax4_fd.legend(loc="best", fontsize=9)
    ax4_fd.grid(True, alpha=0.3)
    ax4_fd.tick_params(axis="x", labelbottom=False)
    ax4_res = fig.add_subplot(gs_right_bot[1])
    ax4_res.plot(phi_depol, np.abs(fd_res_depol), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax4_res.plot(phi_depol, fd_res_depol.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax4_res.plot(phi_depol, fd_res_depol.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax4_res.axhline(sig * sigma_res_depol, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax4_res.axhline(-sig * sigma_res_depol, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax4_res.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax4_res.axvline(peak_phi_d, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax4_res.set_xlim(xlim_phi[0], xlim_phi[1])
    ax4_res.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax4_res.set_ylabel("Residuals", fontsize=11)
    ax4_res.grid(True, alpha=0.3)

    plt.suptitle(f"{source_type} Source: Reference vs Depolarized ({band_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 80)
    print("CS-ROMER Faraday revision script (thin/thick/mixed, SKA bands)")
    print("=" * 80)
    print(f"Reconstructor: {RECONSTRUCTOR}")

    # Faraday grid used implicitly by CSROMERReconstructorWrapper via Parameter.calculate_cellsize,
    # so PHI_MAX / PHI_CELLSIZE are mostly for reference and consistency with testing_faraday.

    for band_name, cfg in SKA_BANDS.items():
        print("\n" + "=" * 80)
        print(f"Processing band: {band_name}")
        print("=" * 80)

        nu = cfg["freq"]()
        short = cfg["short"]
        # SKA-LOW: narrower φ range for Faraday spectrum; other bands use ±PHI_MAX
        phi_xlim = 100.0 if band_name == "SKA-LOW" else None

        # Simulate sources (thin / thick / mixed, clean / RFI / depol)
        sims = simulate_sources_for_band(nu, band_name)

        # THIN: Clean vs RFI and Clean vs Depolarization (same as testing_faraday)
        print("  Reconstructing thin sources...")
        recon_thin_clean = run_csromer_reconstruction(sims["thin_clean"], reconstructor=RECONSTRUCTOR)
        recon_thin_rfi = run_csromer_reconstruction(sims["thin_rfi"], reconstructor=RECONSTRUCTOR)
        recon_thin_depol = run_csromer_reconstruction(sims["thin_depol"], reconstructor=RECONSTRUCTOR)

        print("    Plot: Thin clean vs RFI...")
        plot_2x2_clean_vs_rfi(
            sims["thin_clean"],
            sims["thin_rfi"],
            recon_thin_clean,
            recon_thin_rfi,
            band_label=band_name,
            source_type="Thin",
            filename=f"thin_clean_vs_rfi_{short}.png",
            phi_xlim=phi_xlim,
        )
        print("    Plot: Thin clean vs depolarized...")
        plot_2x2_clean_vs_depol(
            sims["thin_clean"],
            sims["thin_depol"],
            recon_thin_clean,
            recon_thin_depol,
            band_label=band_name,
            source_type="Thin",
            filename=f"thin_depolarization_{short}.png",
            phi_xlim=phi_xlim,
        )

        # THICK + MIXED (skip for SKA-LOW, same as testing_faraday)
        if band_name != "SKA-LOW":
            print("  Reconstructing thick sources...")
            recon_thick_clean = run_csromer_reconstruction(sims["thick_clean"], reconstructor=RECONSTRUCTOR)
            recon_thick_rfi = run_csromer_reconstruction(sims["thick_rfi"], reconstructor=RECONSTRUCTOR)
            recon_thick_depol = run_csromer_reconstruction(sims["thick_depol"], reconstructor=RECONSTRUCTOR)

            print("    Plot: Thick clean vs RFI...")
            plot_2x2_clean_vs_rfi(
                sims["thick_clean"],
                sims["thick_rfi"],
                recon_thick_clean,
                recon_thick_rfi,
                band_label=band_name,
                source_type="Thick",
                filename=f"thick_clean_vs_rfi_{short}.png",
                phi_xlim=phi_xlim,
            )
            print("    Plot: Thick clean vs depolarized...")
            plot_2x2_clean_vs_depol(
                sims["thick_clean"],
                sims["thick_depol"],
                recon_thick_clean,
                recon_thick_depol,
                band_label=band_name,
                source_type="Thick",
                filename=f"thick_depolarization_{short}.png",
                phi_xlim=phi_xlim,
            )

            print("  Reconstructing mixed sources...")
            recon_mixed_clean = run_csromer_reconstruction(sims["mixed_clean"], reconstructor=RECONSTRUCTOR)
            recon_mixed_rfi = run_csromer_reconstruction(sims["mixed_rfi"], reconstructor=RECONSTRUCTOR)

            print("    Plot: Mixed clean vs RFI...")
            plot_2x2_clean_vs_rfi(
                sims["mixed_clean"],
                sims["mixed_rfi"],
                recon_mixed_clean,
                recon_mixed_rfi,
                band_label=band_name,
                source_type="Mixed",
                filename=f"mixed_clean_vs_rfi_{short}.png",
                phi_xlim=phi_xlim,
            )

    print("\nAll 2×2 comparison figures generated (Clean vs RFI, Clean vs Depolarized).")


if __name__ == "__main__":
    # Ensure src is on path when running this script directly
    repo_root = Path(__file__).parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    main()
