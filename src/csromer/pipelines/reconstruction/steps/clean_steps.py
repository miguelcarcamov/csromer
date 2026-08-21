"""
Pipeline step for 1D CLEAN (φ-space Högbom or major-cycle).

Thin adapter: resolve threshold, call ``clean_1d`` / ``clean_1d_major_cycle``,
write results onto the reconstructor context.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from csromer.utils.array_utils import asnumpy

from ..clean import clean_1d, clean_1d_major_cycle
from ..reconstruction_stats import calculate_fd_signal_noise, calculate_second_moment

_KIND_ALIASES = {
    "phi": "phi",
    "phi_space": "phi",
    "hogbom": "phi",
    "rmtf": "phi",
    "major_cycle": "major_cycle",
    "major": "major_cycle",
    "data": "major_cycle",
    "lambda2": "major_cycle",
    "visibility": "major_cycle",
}


def _normalize_kind(kind: str) -> str:
    key = (kind or "phi").strip().lower().replace("-", "_")
    if key not in _KIND_ALIASES:
        raise ValueError("clean kind must be 'phi' or 'major_cycle'; got %r" % (kind, ))
    return _KIND_ALIASES[key]


class Clean1DStep:
    """
    Run CLEAN and store ``fd_model`` / ``fd_residual`` / ``model_data`` on ctx.

    Parameters
    ----------
    kind :
        ``"phi"`` — Högbom in Faraday depth (shifted RMTF).
        ``"major_cycle"`` — predict with A, subtract in λ², dirty via AᴴW.
    """

    def __init__(
        self,
        gain: float = 0.2,
        maxiter: int = 500,
        threshold: Optional[float] = None,
        n_sigma: Optional[float] = None,
        kind: str = "phi",
        verbose: bool = True,
    ):
        self.gain = gain
        self.maxiter = maxiter
        self.threshold = threshold
        self.n_sigma = n_sigma
        self.kind = _normalize_kind(kind)
        self.verbose = verbose

    def _resolve_threshold(self, ctx, fd_dirty: np.ndarray) -> Optional[float]:
        threshold = self.threshold
        if threshold is None and self.n_sigma is not None and self.n_sigma > 0:
            noise_fd = getattr(ctx, "sigma_fd", None)
            if noise_fd is None or noise_fd <= 0:
                noise_fd = calculate_fd_signal_noise(
                    ctx.fd_dirty,
                    ctx.parameter.phi,
                    ctx.parameter.max_faraday_depth,
                )
            threshold = float(self.n_sigma * noise_fd)

        if self.verbose:
            label = "CLEAN-phi" if self.kind == "phi" else "CLEAN-major"
            peak0 = float(np.abs(fd_dirty).max())
            if threshold is None:
                print(
                    "  [%s] no threshold  dirty_peak=%.6e  maxiter=%d" %
                    (label, peak0, self.maxiter)
                )
            else:
                tag = (
                    "%.1f*sigma_fd" % self.n_sigma if
                    (self.n_sigma is not None and self.n_sigma > 0) else "absolute"
                )
                status = "stop (peak < thresh)" if peak0 < threshold else "iterating"
                print(
                    "  [%s] threshold=%.6e (%s)  dirty_peak=%.6e  %s" %
                    (label, threshold, tag, peak0, status)
                )
        return threshold

    def run(self, ctx) -> None:
        fd_dirty = np.asarray(asnumpy(ctx.fd_dirty), dtype=np.complex64)
        threshold = self._resolve_threshold(ctx, fd_dirty)

        if self.kind == "phi":
            rmtf = np.asarray(asnumpy(ctx.measurement_operator.RMTF(0.0)), dtype=np.complex64)
            model, residual = clean_1d(
                fd_dirty,
                rmtf,
                gain=self.gain,
                maxiter=self.maxiter,
                threshold=threshold,
                n_phi=ctx.parameter.n,
            )
        else:
            op = ctx.measurement_operator
            model, residual = clean_1d_major_cycle(
                data=np.asarray(asnumpy(ctx.dataset.data), dtype=np.complex64),
                forward=op.forward,
                dirty_spectrum=op.dirty_spectrum,
                gain=self.gain,
                maxiter=self.maxiter,
                threshold=threshold,
                dirty=fd_dirty,
            )

        ctx.fd_model = model
        ctx.fd_residual = residual
        ctx.parameter.data = model
        ctx.dataset.model_data = ctx.measurement_operator.forward(model)
        ctx.rm_model = ctx.get_rm(model)
        ctx.second_moment = calculate_second_moment(ctx.parameter.phi, model)


def make_clean_1d_step(
    kind: str = "phi",
    gain: float = 0.2,
    maxiter: int = 500,
    threshold: Optional[float] = None,
    n_sigma: Optional[float] = None,
    verbose: bool = True,
) -> Clean1DStep:
    """Build a ``Clean1DStep`` (``kind``: ``"phi"`` or ``"major_cycle"``)."""
    return Clean1DStep(
        gain=gain,
        maxiter=maxiter,
        threshold=threshold,
        n_sigma=n_sigma,
        kind=kind,
        verbose=verbose,
    )
