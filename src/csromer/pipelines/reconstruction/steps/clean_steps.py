"""
Pipeline step for 1D CLEAN: produce fd_model from fd_dirty (no optimization).
"""
from __future__ import annotations

import numpy as np

from csromer.utils.array_utils import asnumpy

from ..clean import clean_1d
from ..reconstruction_stats import calculate_fd_signal_noise, calculate_second_moment


class Clean1DStep:
    """
    1D CLEAN from dirty map: find peaks, add components, subtract scaled RMTF.

    All in Faraday depth space: residual = dirty - (RMTF ⊗ model) from the
    CLEAN loop. Sets ctx.fd_model and ctx.fd_residual (FD-space residual) so
    RestorationStep uses this residual and does not recompute from data space.
    Also sets ctx.dataset.model_data for compatibility.
    """

    def __init__(
        self,
        gain: float = 0.2,
        maxiter: int = 500,
        threshold: float | None = None,
        n_sigma: float | None = None,
        verbose: bool = True,
    ):
        self.gain = gain
        self.maxiter = maxiter
        self.threshold = threshold
        self.n_sigma = n_sigma
        self.verbose = verbose

    def run(self, ctx) -> None:
        fd_dirty = np.asarray(asnumpy(ctx.fd_dirty), dtype=np.complex128)
        n_phi = ctx.parameter.n
        rmtf0 = np.asarray(
            asnumpy(ctx.measurement_operator.RMTF(0.0)), dtype=np.complex128
        )
        rmtf_peak = np.abs(rmtf0).max()
        if rmtf_peak <= 0:
            ctx.fd_model = np.zeros_like(fd_dirty, dtype=np.complex64)
            ctx.fd_residual = np.asarray(fd_dirty, dtype=np.complex64)
            ctx.parameter.data = ctx.fd_model
            ctx.dataset.model_data = ctx.measurement_operator.forward(ctx.fd_model)
            ctx.rm_model = ctx.get_rm(ctx.fd_model)
            ctx.second_moment = 0.0
            return
        rmtf_norm = rmtf0 / rmtf_peak

        # Threshold in Faraday depth space: stop when max(|residual|) < threshold.
        # Use propagated σ_fd (ctx.sigma_fd) when set, else estimate from dirty map edges.
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
        if self.verbose and threshold is not None:
            peak0 = float(np.abs(fd_dirty).max())
            n_sigma_str = "%.1f*sigma_fd" % self.n_sigma if (self.n_sigma is not None and self.n_sigma > 0) else "absolute"
            print(
                "  [CLEAN] threshold=%.6e (%s)  dirty_peak=%.6e  %s"
                % (
                    threshold,
                    n_sigma_str,
                    peak0,
                    "stop (peak < thresh)" if peak0 < threshold else "iterating",
                )
            )

        model, residual = clean_1d(
            fd_dirty,
            rmtf_norm,
            gain=self.gain,
            maxiter=self.maxiter,
            threshold=threshold,
            n_phi=n_phi,
        )
        ctx.fd_model = model.astype(np.complex64)
        ctx.fd_residual = residual.astype(np.complex64)  # FD-space residual from CLEAN loop
        ctx.parameter.data = ctx.fd_model
        ctx.dataset.model_data = ctx.measurement_operator.forward(ctx.fd_model)
        ctx.rm_model = ctx.get_rm(ctx.fd_model)
        ctx.second_moment = calculate_second_moment(ctx.parameter.phi, ctx.fd_model)
