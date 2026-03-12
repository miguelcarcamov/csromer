"""
CS-ROMER Faraday revision script: SKA bands, thin/thick/mixed sources,
RFI and depolarization, plus 1×2 figures:

1. Data in lambda² (polarization vs λ², irregular space — same style as testing_faraday.py)
2. Faraday depth spectrum: dirty, restored, and model on the same plot

Uses csromer simulation classes and CSROMERReconstructorWrapper; saves 1×2 PNGs per scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as C_LIGHT

from csromer.simulation import FaradayThickSource, FaradayThinSource
from csromer.utils.array_utils import maybe_compute
from csromer.wrappers.reconstructors import CSROMERReconstructorWrapper, CGReconstructorWrapper

# Speed of light in m/s (float)
c = float(C_LIGHT.value)


# Colour palette (same idea as testing_faraday)
COLORS = {
    "blue": "#0066CC",
    "orange": "#FF6600",
    "purple": "#9933FF",
    "cyan": "#00CCCC",
    "magenta": "#CC0066",
    "teal": "#009999",
    "yellow": "#FFCC00",
    "black": "#000000",
    "gray": "#666666",
}


# Matplotlib defaults
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.figsize"] = (10, 8)


# SKA-like frequency bands (values copied from testing_faraday.py)
# Using numpy here; csromer simulation classes accept nu arrays.
def ska_low_freq():
    # 50–350 MHz, 3.9 kHz step
    return np.arange(50e6, 350e6, 3.9e3, dtype=np.float64)


def ska_mid_b2_freq():
    # 950–1760 MHz, 13.44 kHz step
    return np.arange(950e6, 1760e6, 13.44e3, dtype=np.float64)


def ska_mid_b5a_freq():
    # 4.6–8.5 GHz, 13.44 kHz step
    return np.arange(4.6e9, 8.5e9, 13.44e3, dtype=np.float64)


def ska_mid_b5b_freq():
    # 8.3–15.4 GHz, 13.44 kHz step
    return np.arange(8.3e9, 15.4e9, 13.44e3, dtype=np.float64)


SKA_BANDS = {
    "SKA-LOW": {"freq": ska_low_freq, "short": "LOW"},
    "SKA-MID B2": {"freq": ska_mid_b2_freq, "short": "B2"},
    "SKA-MID B5a": {"freq": ska_mid_b5a_freq, "short": "B5a"},
    "SKA-MID B5b": {"freq": ska_mid_b5b_freq, "short": "B5b"},
}


# Source / effect parameters (from testing_faraday main block)
THIN_PARAMS = {
    "phi_gal": 15.0,  # rad/m²
    "s_nu": 1.0,
    "spectral_idx": -0.7,
    "dchi": 0.0,
}

THICK_PARAMS = {
    "phi_fg": 10.0,  # rad/m²
    "phi_center": 20.0,  # rad/m²
    "s_nu": 1.0,
    "spectral_idx": -0.7,
}

MIXED_CONFIG = [
    {
        "type": "thin",
        "phi_gal": -400.0,
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

RFI_REMOVE_FRAC = 0.10  # 10% channels removed
DEPOL_SIGMA_RM_THIN = 5.0  # rad/m²
DEPOL_SIGMA_RM_THICK = 3.0  # rad/m²

# Reconstructor: "csromer" (FISTA + L1) or "cg" (conjugate gradient)
RECONSTRUCTOR = "cg"

# Faraday grid parameters (same for all bands; tweak as needed)
PHI_MAX = 1000.0  # rad/m²
PHI_CELLSIZE = 0.5  # rad/m²


def simulate_sources_for_band(nu: np.ndarray, band_name: str):
    """Simulate thin/thick/mixed, clean/RFI/depol for a given band."""
    # Thin clean
    thin_clean = FaradayThinSource(nu=nu, **THIN_PARAMS)
    thin_clean.simulate()

    # Thin RFI
    thin_rfi = FaradayThinSource(nu=nu, **THIN_PARAMS)
    thin_rfi.simulate()
    thin_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(42))

    # Thin depolarized
    thin_depol = FaradayThinSource(nu=nu, **THIN_PARAMS)
    thin_depol.simulate()
    thin_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THIN)

    thick_clean = thick_rfi = thick_depol = None
    mixed_clean = mixed_rfi = None

    # For SKA-LOW we only use thin sources (as in testing_faraday)
    if band_name != "SKA-LOW":
        # Thick clean
        thick_clean = FaradayThickSource(nu=nu, **THICK_PARAMS)
        thick_clean.simulate()

        # Thick RFI
        thick_rfi = FaradayThickSource(nu=nu, **THICK_PARAMS)
        thick_rfi.simulate()
        thick_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(43))

        # Thick depolarized
        thick_depol = FaradayThickSource(nu=nu, **THICK_PARAMS)
        thick_depol.simulate()
        thick_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THICK)

        # Mixed clean
        cfg_thin = {k: v for k, v in MIXED_CONFIG[0].items() if k != "type"}
        cfg_thick = {k: v for k, v in MIXED_CONFIG[1].items() if k != "type"}

        mixed_clean_thin = FaradayThinSource(nu=nu, **cfg_thin)
        mixed_clean_thin.simulate()
        mixed_clean_thick = FaradayThickSource(nu=nu, **cfg_thick)
        mixed_clean_thick.simulate()
        mixed_clean = mixed_clean_thin + mixed_clean_thick

        # Mixed RFI
        mixed_rfi_thin = FaradayThinSource(nu=nu, **cfg_thin)
        mixed_rfi_thin.simulate()
        mixed_rfi_thick = FaradayThickSource(nu=nu, **cfg_thick)
        mixed_rfi_thick.simulate()
        mixed_rfi = mixed_rfi_thin + mixed_rfi_thick
        mixed_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(44))

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
    maxiter: int = 500,
    reconstructor: str = "csromer",
):
    """Run reconstruction on a single csromer Dataset and return recon object.

    Args:
        source: Dataset (e.g. simulated source).
        oversampling: Oversampling factor for Faraday depth grid.
        maxiter: Maximum iterations (FISTA or CG depending on reconstructor).
        reconstructor: "csromer" (FISTA + L1) or "cg".

    Returns:
        Reconstructor instance after reconstruct().
    """
    if reconstructor.lower() == "cg":
        recon = CGReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            cg_maxiter=maxiter,
            cg_tol=1e-12,
            cg_verbose=True,
            lambda_l_norm=1e-10,
            fourier_mode="gridded",
        )
    else:
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            fista_maxiter=maxiter,
            fista_tol=1e-12,
            fista_verbose=True,
            lambda_l_norm=0.0005,
            fourier_mode="gridded",
        )
    recon.reconstruct()
    return recon


def plot_1x2_csromer(
    source,
    recon,
    band_label: str,
    scenario_label: str,
    filename: str | None = None,
    phi_xlim: float | tuple[float, float] | None = None,
):
    """
    Make 1×2 plot:
    - Left: Polarization vs λ² (data before gridding; markers like testing_faraday).
    - Right: Two stacked panels — upper: dirty and restored with 5σ limit; lower: residuals (|res|, Re, Im) with 2σ, 3σ, 5σ. Red vertical line at peak φ.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    # Original (irregular) lambda² and data from simulation
    lambda2 = np.asarray(maybe_compute(source.lambda2))
    P = np.asarray(maybe_compute(source.data))

    phi = np.asarray(maybe_compute(recon.parameter.phi))
    fd_dirty = np.asarray(maybe_compute(recon.fd_dirty))
    fd_restored = np.asarray(maybe_compute(recon.fd_restored))
    fd_residual = np.asarray(maybe_compute(recon.fd_residual))

    # Noise: use edge regions only (threshold=0.5 so |phi| > 0.5*max_fd) to avoid including the source
    sigma_spectrum = recon.calculate_fd_signal_noise(
        recon.fd_dirty,
        recon.parameter.phi,
        recon.parameter.max_faraday_depth,
        threshold=0.5,
    )
    sigma_spectrum = float(np.asarray(maybe_compute(sigma_spectrum)))
    sigma_residual = recon.calculate_fd_signal_noise(
        recon.fd_residual,
        recon.parameter.phi,
        recon.parameter.max_faraday_depth,
        threshold=0.5,
    )
    sigma_residual = float(np.asarray(maybe_compute(sigma_residual)))

    # Peak Faraday depth for red vertical line (use restored peak)
    peak_idx = np.argmax(np.abs(fd_restored))
    phi_peak = float(phi[peak_idx])

    xlim_left = (-PHI_MAX, PHI_MAX)
    if phi_xlim is None:
        xlim_right = (-PHI_MAX, PHI_MAX)
    elif isinstance(phi_xlim, (tuple, list)) and len(phi_xlim) == 2:
        xlim_right = (float(phi_xlim[0]), float(phi_xlim[1]))
    else:
        x = float(phi_xlim)
        xlim_right = (-x, x)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.6], width_ratios=[1, 1], hspace=0)

    # Left: Polarization vs λ² (full height)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_left.plot(lambda2, np.abs(P), ".", color=COLORS["blue"], markersize=0.6, alpha=0.9, label=r"$|P|$")
    ax_left.plot(lambda2, P.real, ".", color=COLORS["purple"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax_left.plot(lambda2, P.imag, ".", color=COLORS["orange"], markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax_left.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax_left.set_ylabel("Polarization [Jy]", fontsize=11)
    ax_left.set_title(f"{scenario_label}: Polarization vs " + r"$\lambda^2$", fontsize=12, fontweight="bold")
    ax_left.legend(loc="best", fontsize=9)
    ax_left.grid(True, alpha=0.3)

    # Right upper: Dirty and restored only; 5σ limit; red line at peak
    ax_upper = fig.add_subplot(gs[0, 1])
    ax_upper.plot(phi, np.abs(fd_dirty), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax_upper.plot(phi, np.abs(fd_restored), "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax_upper.axhline(5.0 * sigma_spectrum, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    ax_upper.axvline(phi_peak, color="red", linestyle="-", lw=1.2, alpha=0.45)
    ax_upper.set_xlim(xlim_right[0], xlim_right[1])
    ax_upper.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax_upper.set_title("Faraday depth spectrum", fontsize=12, fontweight="bold")
    # Legend including peak position (proxy line for the red vertical line)
    h, _ = ax_upper.get_legend_handles_labels()
    peak_handle = Line2D([0], [0], color="red", alpha=0.45, lw=1.2, label=rf"Peak $\phi$ = {phi_peak:.2f}")
    ax_upper.legend(handles=h + [peak_handle], loc="best", fontsize=9)
    ax_upper.grid(True, alpha=0.3)
    ax_upper.tick_params(axis="x", labelbottom=False)

    # Right lower: Residuals (abs, real, imag) and 2σ, 3σ, 5σ limits — no legend
    ax_lower = fig.add_subplot(gs[1, 1])
    ax_lower.plot(phi, np.abs(fd_residual), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax_lower.plot(phi, fd_residual.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax_lower.plot(phi, fd_residual.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax_lower.axhline(sig * sigma_residual, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax_lower.axhline(-sig * sigma_residual, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax_lower.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax_lower.axvline(phi_peak, color="red", linestyle="-", lw=1.2, alpha=0.45)
    ax_lower.set_xlim(xlim_right[0], xlim_right[1])
    ax_lower.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax_lower.set_ylabel("Residuals", fontsize=11)
    ax_lower.grid(True, alpha=0.3)
    ax_lower.tick_params(axis="x", labelbottom=True)
    plt.setp(ax_lower.get_xticklabels(), visible=True)

    plt.suptitle(f"{band_label} — {scenario_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename is not None:
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

        # THIN: clean, RFI, depol
        print("  Reconstructing thin sources...")
        recon_thin_clean = run_csromer_reconstruction(sims["thin_clean"], reconstructor=RECONSTRUCTOR)
        plot_1x2_csromer(
            sims["thin_clean"],
            recon_thin_clean,
            band_label=band_name,
            scenario_label="Thin clean",
            filename=f"thin_clean_1x2_{short}.png",
            phi_xlim=phi_xlim,
        )

        recon_thin_rfi = run_csromer_reconstruction(sims["thin_rfi"], reconstructor=RECONSTRUCTOR)
        plot_1x2_csromer(
            sims["thin_rfi"],
            recon_thin_rfi,
            band_label=band_name,
            scenario_label="Thin RFI",
            filename=f"thin_rfi_1x2_{short}.png",
            phi_xlim=phi_xlim,
        )

        recon_thin_depol = run_csromer_reconstruction(sims["thin_depol"], reconstructor=RECONSTRUCTOR)
        plot_1x2_csromer(
            sims["thin_depol"],
            recon_thin_depol,
            band_label=band_name,
            scenario_label="Thin depol",
            filename=f"thin_depol_1x2_{short}.png",
            phi_xlim=phi_xlim,
        )

        # THICK + MIXED (skip for SKA-LOW)
        if band_name != "SKA-LOW":
            print("  Reconstructing thick sources...")
            recon_thick_clean = run_csromer_reconstruction(sims["thick_clean"], reconstructor=RECONSTRUCTOR)
            plot_1x2_csromer(
                sims["thick_clean"],
                recon_thick_clean,
                band_label=band_name,
                scenario_label="Thick clean",
                filename=f"thick_clean_1x2_{short}.png",
                phi_xlim=phi_xlim,
            )

            recon_thick_rfi = run_csromer_reconstruction(sims["thick_rfi"], reconstructor=RECONSTRUCTOR)
            plot_1x2_csromer(
                sims["thick_rfi"],
                recon_thick_rfi,
                band_label=band_name,
                scenario_label="Thick RFI",
                filename=f"thick_rfi_1x2_{short}.png",
                phi_xlim=phi_xlim,
            )

            recon_thick_depol = run_csromer_reconstruction(sims["thick_depol"], reconstructor=RECONSTRUCTOR)
            plot_1x2_csromer(
                sims["thick_depol"],
                recon_thick_depol,
                band_label=band_name,
                scenario_label="Thick depol",
                filename=f"thick_depol_1x2_{short}.png",
                phi_xlim=phi_xlim,
            )

            print("  Reconstructing mixed sources...")
            recon_mixed_clean = run_csromer_reconstruction(sims["mixed_clean"], reconstructor=RECONSTRUCTOR)
            plot_1x2_csromer(
                sims["mixed_clean"],
                recon_mixed_clean,
                band_label=band_name,
                scenario_label="Mixed clean",
                filename=f"mixed_clean_1x2_{short}.png",
                phi_xlim=phi_xlim,
            )

            recon_mixed_rfi = run_csromer_reconstruction(sims["mixed_rfi"], reconstructor=RECONSTRUCTOR)
            plot_1x2_csromer(
                sims["mixed_rfi"],
                recon_mixed_rfi,
                band_label=band_name,
                scenario_label="Mixed RFI",
                filename=f"mixed_rfi_1x2_{short}.png",
                phi_xlim=phi_xlim,
            )

    print("\nAll 1×2 CS-ROMER figures generated.")


if __name__ == "__main__":
    # Ensure src is on path when running this script directly
    repo_root = Path(__file__).parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    main()
