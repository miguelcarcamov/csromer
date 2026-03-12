"""
Reconstruction statistics: noise estimation, peak interpolation, RM errors.

Pure functions used by the CS-ROMER reconstructor. No dependency on the
reconstructor class; all inputs are passed explicitly.
"""
from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats


def estimate_peak_quadratic_interpolation(
    fd_signal: np.ndarray, cellsize: float
) -> tuple[float, float]:
    """
    Estimate peak location and value using quadratic interpolation.

    Fits quadratic to peak and neighbors for sub-pixel accuracy.

    Returns:
        (phi_peak, peak_value)
    """
    length_n = len(fd_signal)
    index_0 = int(np.argmax(np.abs(fd_signal)))

    if index_0 <= 0 or index_0 >= length_n - 1:
        location = float(index_0)
        estimated_peak = float(np.abs(fd_signal[index_0]))
        pos_phi_peak = (location - length_n / 2) * cellsize
        return pos_phi_peak, estimated_peak

    fd_signal_0 = np.abs(fd_signal[index_0])
    fd_signal_m1 = np.abs(fd_signal[index_0 - 1])
    fd_signal_p1 = np.abs(fd_signal[index_0 + 1])

    pos_estimated_peak = (fd_signal_p1 - fd_signal_m1) / (
        4 * fd_signal_0 - 2 * fd_signal_m1 - 2 * fd_signal_p1
    )
    estimated_peak = fd_signal_0 - 0.25 * (fd_signal_m1 - fd_signal_p1) * pos_estimated_peak
    location = index_0 + pos_estimated_peak
    pos_phi_peak = (location - length_n / 2) * cellsize
    return pos_phi_peak, estimated_peak


def calculate_ricean_peak(peak: float, noise: float) -> float:
    """Ricean-corrected peak value."""
    return float(np.sqrt(peak**2 - (2.3 * noise**2)))


def _robust_rms(data, sigma_val, cenfunc_val, stdfunc_val):
    """Robust RMS with sigma clipping; fallback to std on failure."""
    n_points = len(data)
    if n_points < 10:
        return np.std(data) if n_points > 1 else 0.0
    initial_std = np.std(data)
    initial_mean = np.mean(data)
    data_range = np.max(data) - np.min(data)
    if initial_std < 1e-10 or data_range < 1e-10:
        return initial_std
    if abs(initial_mean) > 1e-10:
        cv = initial_std / abs(initial_mean)
        if cv < 1e-6:
            return initial_std
    try:
        _, _, rms = sigma_clipped_stats(
            data, sigma=sigma_val, cenfunc=cenfunc_val, stdfunc=stdfunc_val, maxiters=1
        )
        if np.isfinite(rms) and rms > 0 and rms <= initial_std * 10:
            return rms
    except (ValueError, RuntimeError):
        pass
    return initial_std


def calculate_fd_signal_noise(
    fd_signal: np.ndarray,
    phi: np.ndarray,
    max_fd_depth: float,
    threshold: float = 0.0,
    sigma: float = 0.3,
    cenfunc: str = "mean",
    stdfunc: str = "mad_std",
) -> float:
    """
    Estimate noise in Faraday depth signal from edge regions.

    Uses sigma-clipped statistics on |phi| > max_fd_depth * threshold.
    """
    edge_mask = np.abs(phi) > max_fd_depth * threshold
    n_edge_points = np.sum(edge_mask)
    for frac in (0.5, 0.2, 0.1):
        if n_edge_points < 10:
            edge_mask = np.abs(phi) > max_fd_depth * frac
            n_edge_points = np.sum(edge_mask)
    if n_edge_points < 10:
        edge_mask = np.ones_like(phi, dtype=bool)

    # Work component-wise and combine via variance propagation rather than
    # a simple arithmetic mean. This preserves the use of a robust,
    # MAD-based estimator (default stdfunc="mad_std") and gives a single
    # scalar sigma for the complex Faraday spectrum.
    edge_real = fd_signal.real[edge_mask]
    edge_imag = fd_signal.imag[edge_mask]

    rms_real = _robust_rms(edge_real, sigma, cenfunc, stdfunc)
    rms_imag = _robust_rms(edge_imag, sigma, cenfunc, stdfunc)

    # Fallback: retry with MAD-based robust RMS explicitly if the first
    # pass failed or returned ~0. Avoid dropping back to plain np.std.
    if not np.isfinite(rms_real) or rms_real == 0:
        rms_real = _robust_rms(edge_real, sigma, cenfunc, "mad_std")
    if not np.isfinite(rms_imag) or rms_imag == 0:
        rms_imag = _robust_rms(edge_imag, sigma, cenfunc, "mad_std")

    # If we *still* fail, use a tiny fraction of the peak as a last resort.
    if (not np.isfinite(rms_real)) or rms_real == 0:
        rms_real = np.max(np.abs(fd_signal.real)) * 1e-6
    if (not np.isfinite(rms_imag)) or rms_imag == 0:
        rms_imag = np.max(np.abs(fd_signal.imag)) * 1e-6

    # Proper combination for a complex quantity: average the variances of
    # Re and Im and then take the square root. When rms_real ~= rms_imag,
    # this reduces to that common sigma, unlike 0.5*(rms_real + rms_imag).
    fd_signal_noise = float(np.sqrt(0.5 * (rms_real**2 + rms_imag**2)))

    if fd_signal_noise == 0 or not np.isfinite(fd_signal_noise):
        fd_signal_noise = np.max(np.abs(fd_signal)) * 1e-6
    return float(fd_signal_noise)


def calculate_sigma_phi_peak(
    rmtf_fwhm: float, fd_peak: float, fd_signal_noise: float
) -> float:
    """Error on RM peak (rad/m²). Returns NaN if peak or noise is zero."""
    denom = 2.0 * fd_peak
    if denom == 0 or fd_signal_noise == 0:
        return np.nan
    return float(rmtf_fwhm * fd_signal_noise / denom)


def calculate_second_moment(phi: np.ndarray, fd_model: np.ndarray) -> float:
    """Weighted second moment of model around first moment (rad²/m⁴)."""
    mask = np.abs(fd_model) != 0
    phi_nz = phi[mask]
    fd_nz = fd_model[mask]
    fd_abs = np.abs(fd_nz)
    k = np.sum(fd_abs)
    if k == 0 or phi_nz.size == 0:
        return 0.0
    first = np.sum(phi_nz * fd_abs) / k
    return float(np.sum(fd_abs * (phi_nz - first) ** 2) / k)
