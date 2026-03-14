"""
1D CLEAN algorithm for Faraday depth.

Pure implementation: takes dirty spectrum and RMTF (at phi=0), returns model and
residual. Assumes equispaced phi grid; RMTF is shifted by integer roll for each
component. No dependency on Parameter, Dataset, or measurement operator.
"""
from __future__ import annotations

import numpy as np


def _shift_rmtf_to_peak(rmtf0: np.ndarray, peak_idx: int, n_phi: int) -> np.ndarray:
    """
    Shift RMTF so its peak aligns with peak_idx (for equispaced phi grid).

    RMTF(0) has its peak at center index n_phi // 2. Roll so that peak moves
    to peak_idx.

    Args:
        rmtf0: RMTF array (n_phi,) with peak at center.
        peak_idx: Target index for the peak.
        n_phi: Length of the grid (len(rmtf0)).

    Returns:
        Shifted RMTF, same shape as rmtf0.
    """
    shift = int(peak_idx) - n_phi // 2
    return np.roll(rmtf0, shift)


def clean_1d(
    dirty: np.ndarray,
    rmtf_at_zero: np.ndarray,
    gain: float,
    maxiter: int,
    threshold: float | None = None,
    n_phi: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D CLEAN loop: find peaks in residual, add scaled components to model,
    subtract scaled shifted RMTF from residual.

    Args:
        dirty: Complex 1D dirty Faraday spectrum.
        rmtf_at_zero: Complex 1D RMTF at phi=0, same length as dirty, peak at center.
        gain: Loop gain (typical 0.1--0.3).
        maxiter: Maximum number of CLEAN components.
        threshold: Optional; stop when max(|residual|) < threshold.
        n_phi: Length of grid (default len(dirty)); used for shift.

    Returns:
        (model, residual): Complex 1D arrays, same shape as dirty.
    """
    dirty = np.asarray(dirty, dtype=np.complex128)
    rmtf_at_zero = np.asarray(rmtf_at_zero, dtype=np.complex128)
    n = len(dirty)
    if n != len(rmtf_at_zero):
        raise ValueError("dirty and rmtf_at_zero must have the same length")
    n_phi = n if n_phi is None else n_phi

    # Normalise RMTF to peak 1 so subtraction scale is correct
    rmtf_peak = np.abs(rmtf_at_zero).max()
    if rmtf_peak <= 0:
        return np.zeros_like(dirty), dirty.copy()
    rmtf_norm = rmtf_at_zero / rmtf_peak

    model = np.zeros_like(dirty)
    residual = dirty.copy()

    for _ in range(maxiter):
        imax = int(np.argmax(np.abs(residual)))
        peak_val = residual[imax]
        peak_amp = np.abs(peak_val)
        if threshold is not None and peak_amp < threshold:
            break
        if peak_amp <= 0:
            break

        model[imax] += gain * peak_val
        beam = _shift_rmtf_to_peak(rmtf_norm, imax, n_phi)
        residual -= gain * peak_val * beam

    return model.astype(dirty.dtype), residual.astype(dirty.dtype)
