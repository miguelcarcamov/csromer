"""
1D CLEAN for Faraday depth reconstruction.

Two variants with the same greedy peak-picking, different residual updates:

* ``clean_1d`` — Högbom in φ-space: subtract a shifted RMTF from the FD residual.
* ``clean_1d_major_cycle`` — major cycle: predict with ``forward``, subtract in λ²,
  then form the FD residual with ``dirty_spectrum``.

Arrays are ``complex64``, matching Dataset / measurement operators.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _shift_rmtf_to_peak(rmtf0: np.ndarray, peak_idx: int, n_phi: int) -> np.ndarray:
    """Roll RMTF so the peak at index ``n_phi // 2`` moves to ``peak_idx``."""
    return np.roll(rmtf0, int(peak_idx) - n_phi // 2)


def clean_1d(
    dirty: np.ndarray,
    rmtf_at_zero: np.ndarray,
    gain: float,
    maxiter: int,
    threshold: Optional[float] = None,
    n_phi: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    φ-space Högbom CLEAN.

    Each iteration finds the peak of the FD residual, adds ``gain * peak`` as a
    δ-component to the model, and subtracts ``gain * peak * shifted_RMTF``.

    Returns
    -------
    model, residual
        Both ``complex64``, same length as ``dirty``.
    """
    dirty = np.asarray(dirty, dtype=np.complex64)
    rmtf = np.asarray(rmtf_at_zero, dtype=np.complex64)
    if dirty.shape != rmtf.shape:
        raise ValueError("dirty and rmtf_at_zero must have the same length")
    n_phi = len(dirty) if n_phi is None else int(n_phi)

    rmtf_peak = float(np.abs(rmtf).max())
    if rmtf_peak <= 0.0:
        return np.zeros_like(dirty), dirty.copy()
    rmtf = rmtf / rmtf_peak

    model = np.zeros_like(dirty)
    residual = dirty.copy()

    for _ in range(maxiter):
        peak_idx = int(np.argmax(np.abs(residual)))
        peak = residual[peak_idx]
        peak_amp = float(np.abs(peak))
        if peak_amp <= 0.0:
            break
        if threshold is not None and peak_amp < threshold:
            break

        component = np.complex64(gain * peak)
        model[peak_idx] += component
        residual -= component * _shift_rmtf_to_peak(rmtf, peak_idx, n_phi)

    return model, residual


def clean_1d_major_cycle(
    data: np.ndarray,
    forward: Callable[[np.ndarray], np.ndarray],
    dirty_spectrum: Callable[[np.ndarray], np.ndarray],
    gain: float,
    maxiter: int,
    threshold: Optional[float] = None,
    dirty: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Major-cycle CLEAN (residual formed via the measurement operator).

    Each iteration finds the peak of the FD residual, adds ``gain * peak`` as a
    δ-component, then sets::

        residual = dirty_spectrum(data - forward(model))

    ``forward`` and ``dirty_spectrum`` are callables so this module stays free of
    Dataset / Parameter types; pass ``op.forward`` and ``op.dirty_spectrum``.

    Returns
    -------
    model, residual
        Both ``complex64`` Faraday-depth arrays.
    """
    data = np.asarray(data, dtype=np.complex64)
    if dirty is None:
        residual = np.asarray(dirty_spectrum(data), dtype=np.complex64)
    else:
        residual = np.asarray(dirty, dtype=np.complex64).copy()

    model = np.zeros_like(residual)
    if float(np.abs(residual).max()) <= 0.0:
        return model, residual

    for _ in range(maxiter):
        peak_idx = int(np.argmax(np.abs(residual)))
        peak = residual[peak_idx]
        peak_amp = float(np.abs(peak))
        if peak_amp <= 0.0:
            break
        if threshold is not None and peak_amp < threshold:
            break

        model[peak_idx] += np.complex64(gain * peak)
        predicted = np.asarray(forward(model), dtype=np.complex64)
        residual = np.asarray(dirty_spectrum(data - predicted), dtype=np.complex64)

    return model, residual
