"""
2×2 comparison plots: Clean vs RFI, Clean vs Depolarization.
Shared helpers and panel drawing to avoid redundancy.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from faraday_testing.config import COLORS, PHI_MAX, setup_matplotlib


def sigma_from_amplitude(amp: np.ndarray, method: str = "median") -> float:
    """Estimate Gaussian (Ricean) σ from amplitude |F|. method: 'median' or 'rms'."""
    amp = np.asarray(amp)
    if method == "median":
        val = float(np.median(amp))
        if val <= 0.0 or not np.isfinite(val):
            return 0.0
        return val / np.sqrt(2.0 * np.log(2.0))
    if method == "rms":
        rms = float(np.sqrt(np.mean(amp**2)))
        if not np.isfinite(rms) or rms <= 0:
            return 0.0
        return rms / np.sqrt(2.0)
    raise ValueError(f"method must be 'median' or 'rms', got {method!r}")


def peak_legend_label(peak_phi: float, peak_error: float | None) -> str:
    """Format peak φ with optional error for legend (rad/m²)."""
    if peak_error is not None and np.isfinite(peak_error):
        err_str = f"{peak_error:.2f}" if abs(peak_error) >= 1e-2 else f"{peak_error:.2e}"
        return rf"Peak $\phi$ = {peak_phi:.2f} $\pm$ {err_str}"
    return rf"Peak $\phi$ = {peak_phi:.2f}"


def _phi_xlim(phi_xlim, phi_max: float = PHI_MAX) -> tuple[float, float]:
    if phi_xlim is None:
        return (-phi_max, phi_max)
    if isinstance(phi_xlim, (tuple, list)) and len(phi_xlim) == 2:
        return (float(phi_xlim[0]), float(phi_xlim[1]))
    x = float(phi_xlim)
    return (-x, x)


def _draw_pol_vs_l2(ax, l2, data, color_main: str, title: str, use_distinct_re_im: bool = True) -> None:
    ax.plot(l2, np.abs(data), ".", color=color_main, markersize=0.6, alpha=0.9, label=r"$|P|$")
    c_re = COLORS["purple"] if (use_distinct_re_im and color_main == COLORS["blue"]) else color_main
    c_im = COLORS["orange"] if (use_distinct_re_im and color_main == COLORS["blue"]) else color_main
    ax.plot(l2, data.real, ".", color=c_re, markersize=0.5, alpha=0.8, label=r"$\mathrm{Re}(P)$")
    ax.plot(l2, data.imag, ".", color=c_im, markersize=0.5, alpha=0.8, label=r"$\mathrm{Im}(P)$")
    ax.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax.set_ylabel("Polarization [Jy]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def _draw_fd_panel(
    ax,
    phi,
    fd_dirty,
    fd_restored,
    sigma,
    recon,
    xlim_phi: tuple[float, float],
    title: str,
) -> float:
    """Draw FD spectrum panel. Restored curve is |fd_restored| (amplitude of restored complex spectrum)."""
    ax.plot(phi, np.abs(fd_dirty), "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    restored_amp = np.abs(np.asarray(fd_restored))
    ax.plot(phi, restored_amp, "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    ax.axhline(5.0 * sigma, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"5$\sigma$")
    # Prefer quadratic-interpolated RM and its error when available, falling back to grid-based peak.
    peak_phi = getattr(
        recon,
        "rm_restored_quadratic_interpolation",
        getattr(recon, "rm_restored", None),
    )
    if peak_phi is None:
        peak_idx = int(np.argmax(restored_amp))
        peak_phi = float(phi[peak_idx])
    peak_err = getattr(
        recon,
        "rm_restored_quadratic_interpolation_error",
        getattr(recon, "rm_restored_error", None),
    )
    ax.axvline(peak_phi, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45, label=peak_legend_label(peak_phi, peak_err))
    ax.set_xlim(xlim_phi[0], xlim_phi[1])
    ax.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return peak_phi


def _draw_residual_panel(ax, phi, fd_res, sigma_res: float, peak_phi: float, xlim_phi: tuple[float, float]) -> None:
    ax.plot(phi, np.abs(fd_res), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax.plot(phi, fd_res.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax.plot(phi, fd_res.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax.axhline(sig * sigma_res, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
        ax.axhline(-sig * sigma_res, color=COLORS["gray"], linestyle="--", lw=0.8, alpha=0.7)
    ax.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax.axvline(peak_phi, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax.set_xlim(xlim_phi[0], xlim_phi[1])
    ax.set_xlabel(r"$\phi$ [rad/m²]", fontsize=11)
    ax.set_ylabel("Residuals", fontsize=11)
    ax.grid(True, alpha=0.3)


def _build_2x2_figure(figsize=(18, 12)):
    setup_matplotlib()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_right_top = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)
    gs_right_bot = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)
    return fig, gs, gs_right_top, gs_right_bot


def plot_2x2_clean_vs_rfi(
    clean_source,
    rfi_source,
    recon_clean,
    recon_rfi,
    band_label: str,
    source_type: str,
    filename: str | None = None,
    phi_xlim=None,
    figsize=(18, 12),
) -> None:
    """2×2: Clean (top) vs RFI (bottom). Left: pol vs λ²; right: FD spectrum + residuals."""
    xlim_phi = _phi_xlim(phi_xlim)
    l2_c = np.asarray(clean_source.lambda2)
    data_c = np.asarray(clean_source.data)
    l2_r = np.asarray(rfi_source.lambda2)
    data_r = np.asarray(rfi_source.data)

    phi_c = np.asarray(recon_clean.parameter.phi)
    fd_dirty_c = np.asarray(recon_clean.fd_dirty)
    fd_c = np.asarray(recon_clean.fd_restored)
    fd_res_c = np.asarray(recon_clean.fd_residual)
    phi_r = np.asarray(recon_rfi.parameter.phi)
    fd_dirty_r = np.asarray(recon_rfi.fd_dirty)
    fd_r = np.asarray(recon_rfi.fd_restored)
    fd_res_r = np.asarray(recon_rfi.fd_residual)

    sigma_c = sigma_from_amplitude(np.abs(fd_dirty_c), "median")
    sigma_r = sigma_from_amplitude(np.abs(fd_dirty_r), "median")
    sigma_res_c = sigma_from_amplitude(np.abs(fd_res_c), "rms")
    sigma_res_r = sigma_from_amplitude(np.abs(fd_res_r), "rms")

    fig, gs, gs_rt, gs_rb = _build_2x2_figure(figsize)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_pol_vs_l2(ax1, l2_c, data_c, COLORS["blue"], r"Reference (no RFI): Polarization vs $\lambda^2$")

    ax2_fd = fig.add_subplot(gs_rt[0])
    peak_c = _draw_fd_panel(ax2_fd, phi_c, fd_dirty_c, fd_c, sigma_c, recon_clean, xlim_phi, "Reference (no RFI): Faraday depth spectrum")
    ax2_res = fig.add_subplot(gs_rt[1])
    _draw_residual_panel(ax2_res, phi_c, fd_res_c, sigma_res_c, peak_c, xlim_phi)

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_pol_vs_l2(ax3, l2_r, data_r, COLORS["blue"], r"With RFI: Polarization vs $\lambda^2$")

    ax4_fd = fig.add_subplot(gs_rb[0])
    peak_r = _draw_fd_panel(ax4_fd, phi_r, fd_dirty_r, fd_r, sigma_r, recon_rfi, xlim_phi, "With RFI: Faraday depth spectrum")
    ax4_res = fig.add_subplot(gs_rb[1])
    _draw_residual_panel(ax4_res, phi_r, fd_res_r, sigma_res_r, peak_r, xlim_phi)

    fig.suptitle(f"{source_type} Source: Reference vs RFI ({band_label})", fontsize=14, fontweight="bold")
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
    phi_xlim=None,
    figsize=(18, 12),
) -> None:
    """2×2: Clean (top) vs Depolarized (bottom). Left: pol vs λ²; right: FD spectrum + residuals."""
    xlim_phi = _phi_xlim(phi_xlim)
    l2_c = np.asarray(clean_source.lambda2)
    data_c = np.asarray(clean_source.data)
    l2_d = np.asarray(depol_source.lambda2)
    data_d = np.asarray(depol_source.data)

    phi_c = np.asarray(recon_clean.parameter.phi)
    fd_dirty_c = np.asarray(recon_clean.fd_dirty)
    fd_c = np.asarray(recon_clean.fd_restored)
    fd_res_c = np.asarray(recon_clean.fd_residual)
    phi_d = np.asarray(recon_depol.parameter.phi)
    fd_dirty_d = np.asarray(recon_depol.fd_dirty)
    fd_d = np.asarray(recon_depol.fd_restored)
    fd_res_d = np.asarray(recon_depol.fd_residual)

    sigma_c = sigma_from_amplitude(np.abs(fd_dirty_c), "median")
    sigma_d = sigma_from_amplitude(np.abs(fd_dirty_d), "median")
    sigma_res_c = sigma_from_amplitude(np.abs(fd_res_c), "rms")
    sigma_res_d = sigma_from_amplitude(np.abs(fd_res_d), "rms")

    fig, gs, gs_rt, gs_rb = _build_2x2_figure(figsize)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_pol_vs_l2(ax1, l2_c, data_c, COLORS["blue"], r"Reference (no depol.): Polarization vs $\lambda^2$")

    ax2_fd = fig.add_subplot(gs_rt[0])
    peak_c = _draw_fd_panel(ax2_fd, phi_c, fd_dirty_c, fd_c, sigma_c, recon_clean, xlim_phi, "Reference (no depol.): Faraday depth spectrum")
    ax2_res = fig.add_subplot(gs_rt[1])
    _draw_residual_panel(ax2_res, phi_c, fd_res_c, sigma_res_c, peak_c, xlim_phi)

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_pol_vs_l2(ax3, l2_d, data_d, COLORS["orange"], r"Depolarized: Polarization vs $\lambda^2$", use_distinct_re_im=False)

    ax4_fd = fig.add_subplot(gs_rb[0])
    peak_d = _draw_fd_panel(ax4_fd, phi_d, fd_dirty_d, fd_d, sigma_d, recon_depol, xlim_phi, "Depolarized: Faraday depth spectrum")
    ax4_res = fig.add_subplot(gs_rb[1])
    _draw_residual_panel(ax4_res, phi_d, fd_res_d, sigma_res_d, peak_d, xlim_phi)

    fig.suptitle(f"{source_type} Source: Reference vs Depolarized ({band_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
