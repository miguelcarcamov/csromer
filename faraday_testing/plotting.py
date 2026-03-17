"""
2×2 comparison plots: Clean vs RFI, Clean vs Depolarization.
Shared helpers and panel drawing to avoid redundancy.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from faraday_testing.config import COLORS, PHI_MAX, setup_matplotlib


def sigma_from_complex_residual(fd_residual: np.ndarray) -> float:
    """Estimate noise σ from FD residual using real and imaginary parts together.

    For complex Gaussian noise with Re, Im ~ N(0, σ²), we have
    E[Re² + Im²] = 2σ², so σ = sqrt(mean(Re² + Im²) / 2). Uses both components
    rather than amplitude |F|, so the same σ applies to Re, Im, and |F|.
    """
    z = np.asarray(fd_residual, dtype=complex)
    re, im = z.real.ravel(), z.imag.ravel()
    var_both = float(np.mean(re**2 + im**2))
    if not np.isfinite(var_both) or var_both <= 0:
        return 0.0
    return np.sqrt(var_both / 2.0)


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
    if use_distinct_re_im:
        c_re, c_im = COLORS["purple"], COLORS["orange"]
    else:
        c_re = c_im = color_main
    # Draw Re and Im first, then |P| on top so amplitude in color_main is visible.
    ax.plot(l2, data.real, ".", color=c_re, markersize=0.5, alpha=0.8, label="$Q$")
    ax.plot(l2, data.imag, ".", color=c_im, markersize=0.5, alpha=0.8, label="$U$")
    ax.plot(l2, np.abs(data), ".", color=color_main, markersize=0.8, alpha=0.9, label=r"$|P|$")
    ax.set_xlabel(r"$\lambda^2$ [m²]", fontsize=11)
    ax.set_ylabel("Polarization [Jy]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def _sigma_for_both_panels(recon, sigma_residual: float) -> float:
    """Sigma for 2σ/3σ/5σ lines: same in FD (dirty/restored) and residual panels. Uses sigma_fd when set."""
    sigma_fd = getattr(recon, "sigma_fd", None)
    if sigma_fd is not None and sigma_fd > 0:
        return float(sigma_fd)
    return sigma_residual


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
    """Draw FD spectrum panel. sigma is the same as residual panel (sigma_fd when set, else residual-based).
    Y-limits are data-driven so dirty/restored remain visible when they are much smaller than sigma (e.g. model=0).
    """
    fd_dirty = np.asarray(fd_dirty)
    fd_restored = np.asarray(fd_restored)
    dirty_amp = np.abs(fd_dirty)
    restored_amp = np.abs(fd_restored)
    ax.plot(phi, dirty_amp, "-", color=COLORS["teal"], lw=1.2, alpha=0.9, label=r"Dirty $|F(\phi)|$")
    ax.plot(phi, restored_amp, "-", color=COLORS["black"], lw=1.5, alpha=0.9, label=r"Restored $|F(\phi)|$")
    for sig in [2, 3, 5]:
        ax.axhline(sig * sigma, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8, label=r"2$\sigma$, 3$\sigma$, 5$\sigma$" if sig == 2 else None)
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
    # Data-driven ylim so dirty/restored are visible when model=0 (noise only) and sigma lines are large
    data_max = float(np.maximum(dirty_amp.max(), restored_amp.max()))
    y_max = max(1.2 * data_max, 1e-12)
    ax.set_ylim(0, y_max)
    ax.set_ylabel(r"$|F(\phi)|$ [Jy/RMSF]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return peak_phi


def _draw_residual_panel(ax, phi, fd_res, sigma_line: float, peak_phi: float, xlim_phi: tuple[float, float]) -> None:
    """Residual panel. sigma_line must be the same as FD panel (so 2σ/3σ/5σ match).
    Y-limits are data-driven so residual is visible when it is much smaller than sigma (e.g. model=0).
    """
    fd_res = np.asarray(fd_res)
    ax.plot(phi, np.abs(fd_res), "-", color=COLORS["blue"], lw=1, alpha=0.9)
    ax.plot(phi, fd_res.real, "--", color=COLORS["blue"], lw=0.9, alpha=0.8)
    ax.plot(phi, fd_res.imag, ":", color=COLORS["blue"], lw=0.9, alpha=0.8)
    for sig in [2, 3, 5]:
        ax.axhline(sig * sigma_line, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8)
        ax.axhline(-sig * sigma_line, color=COLORS["gray"], linestyle="--", lw=1, alpha=0.8)
    ax.axhline(0, color=COLORS["gray"], linestyle="-", lw=0.5, alpha=0.5)
    ax.axvline(peak_phi, color=COLORS["accent"], linestyle="-", lw=1.2, alpha=0.45)
    ax.set_xlim(xlim_phi[0], xlim_phi[1])
    # Data-driven ylim so residual is visible when model=0 (dirty = residual, small amplitude)
    res_max = float(max(np.abs(fd_res).max(), np.abs(fd_res.real).max(), np.abs(fd_res.imag).max()))
    y_max = max(1.2 * res_max, 1e-12)
    ax.set_ylim(-y_max, y_max)
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
    """2×2: Clean (top) vs RFI (bottom). Left: pol vs λ²; right: FD spectrum + residuals.

    Uses recon.fd_dirty, recon.fd_restored, recon.fd_residual (set by both
    CS-ROMER and CLEAN pipelines; for CLEAN, fd_residual is the FD-space residual).
    """
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

    # Same sigma for FD (dirty/restored) and residual panels: sigma_fd when set, else residual-based.
    sigma_res_c = sigma_from_complex_residual(fd_res_c)
    sigma_res_r = sigma_from_complex_residual(fd_res_r)
    sigma_line_c = _sigma_for_both_panels(recon_clean, sigma_res_c)
    sigma_line_r = _sigma_for_both_panels(recon_rfi, sigma_res_r)

    fig, gs, gs_rt, gs_rb = _build_2x2_figure(figsize)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_pol_vs_l2(ax1, l2_c, data_c, COLORS["blue"], r"Reference (no RFI): Polarization vs $\lambda^2$")

    ax2_fd = fig.add_subplot(gs_rt[0])
    peak_c = _draw_fd_panel(ax2_fd, phi_c, fd_dirty_c, fd_c, sigma_line_c, recon_clean, xlim_phi, "Reference (no RFI): Faraday depth spectrum")
    ax2_res = fig.add_subplot(gs_rt[1])
    _draw_residual_panel(ax2_res, phi_c, fd_res_c, sigma_line_c, peak_c, xlim_phi)

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_pol_vs_l2(ax3, l2_r, data_r, COLORS["blue"], r"With RFI: Polarization vs $\lambda^2$")

    ax4_fd = fig.add_subplot(gs_rb[0])
    peak_r = _draw_fd_panel(ax4_fd, phi_r, fd_dirty_r, fd_r, sigma_line_r, recon_rfi, xlim_phi, "With RFI: Faraday depth spectrum")
    ax4_res = fig.add_subplot(gs_rb[1])
    _draw_residual_panel(ax4_res, phi_r, fd_res_r, sigma_line_r, peak_r, xlim_phi)

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
    """2×2: Clean (top) vs Depolarized (bottom). Left: pol vs λ²; right: FD spectrum + residuals.

    Uses recon.fd_dirty, recon.fd_restored, recon.fd_residual (set by both
    CS-ROMER and CLEAN pipelines; for CLEAN, fd_residual is the FD-space residual).
    """
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

    # Same sigma for FD (dirty/restored) and residual panels: sigma_fd when set, else residual-based.
    sigma_res_c = sigma_from_complex_residual(fd_res_c)
    sigma_res_d = sigma_from_complex_residual(fd_res_d)
    sigma_line_c = _sigma_for_both_panels(recon_clean, sigma_res_c)
    sigma_line_d = _sigma_for_both_panels(recon_depol, sigma_res_d)

    fig, gs, gs_rt, gs_rb = _build_2x2_figure(figsize)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_pol_vs_l2(ax1, l2_c, data_c, COLORS["blue"], r"Reference (no depol.): Polarization vs $\lambda^2$")

    ax2_fd = fig.add_subplot(gs_rt[0])
    peak_c = _draw_fd_panel(ax2_fd, phi_c, fd_dirty_c, fd_c, sigma_line_c, recon_clean, xlim_phi, "Reference (no depol.): Faraday depth spectrum")
    ax2_res = fig.add_subplot(gs_rt[1])
    _draw_residual_panel(ax2_res, phi_c, fd_res_c, sigma_line_c, peak_c, xlim_phi)

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_pol_vs_l2(ax3, l2_d, data_d, COLORS["blue"], r"Depolarized: Polarization vs $\lambda^2$")

    ax4_fd = fig.add_subplot(gs_rb[0])
    peak_d = _draw_fd_panel(ax4_fd, phi_d, fd_dirty_d, fd_d, sigma_line_d, recon_depol, xlim_phi, "Depolarized: Faraday depth spectrum")
    ax4_res = fig.add_subplot(gs_rb[1])
    _draw_residual_panel(ax4_res, phi_d, fd_res_d, sigma_line_d, peak_d, xlim_phi)

    fig.suptitle(f"{source_type} Source: Reference vs Depolarized ({band_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
