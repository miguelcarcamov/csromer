"""
Optimization (CS-ROMER) pipeline steps.

Each step runs with ctx = CSROMERReconstructorWrapper instance.
"""
from __future__ import annotations

import numpy as np

from csromer.utils.array_utils import asnumpy

from ..defaults import (
    build_measurement_operator,
    build_parameter,
    default_objective_factory,
)
from ..optimizer_factories import make_fista_optimizer
from ..reconstruction_stats import (
    calculate_fd_signal_noise,
    calculate_second_moment,
    calculate_sigma_phi_peak,
    estimate_peak_quadratic_interpolation,
)


def _get_rm(ctx, fd_data: np.ndarray) -> float:
    fd_abs = np.asarray(asnumpy(np.abs(fd_data)))
    idx = int(np.argmax(fd_abs))
    phi = np.asarray(asnumpy(ctx.parameter.phi))
    return float(phi[idx])


class L2ZeroStep:
    """Set dataset.l2_ref from calculate_l2ref() when calculate_l2_zero is True."""

    def run(self, ctx) -> None:
        if getattr(ctx, "calculate_l2_zero", False) and ctx.dataset is not None:
            print("Calculating l2_0")
            ctx.dataset.l2_ref = ctx.dataset.calculate_l2ref()


class BuildParameterStep:
    """Build parameter from dataset + oversampling when parameter is None."""

    def run(self, ctx) -> None:
        if getattr(ctx, "parameter", None) is None:
            ctx.parameter = build_parameter(
                ctx.dataset, oversampling=getattr(ctx, "oversampling", 7.0)
            )


class BuildMeasurementOperatorStep:
    """Build measurement operator when None; may replace ctx.dataset for gridded."""

    def run(self, ctx) -> None:
        if getattr(ctx, "measurement_operator", None) is None:
            kind = getattr(ctx, "measurement_operator_kind", "direct")
            # When gridding, store effective_n before replace so we can scale regularization
            if kind == "gridded" and ctx.dataset is not None:
                n_eff = getattr(ctx.dataset, "effective_n", None)
                ctx._n_eff_before_grid = float(n_eff) if n_eff is not None else None
            gridding_kernel = getattr(ctx, "gridding_kernel", "kaiser")
            gridding_kernel_half_width = getattr(ctx, "gridding_kernel_half_width", 4.0)
            gridding_kernel_beta = getattr(ctx, "gridding_kernel_beta", 2.5)
            ctx.measurement_operator, ctx.dataset = build_measurement_operator(
                ctx.dataset,
                ctx.parameter,
                kind,
                gridding_kernel=gridding_kernel,
                gridding_kernel_half_width=gridding_kernel_half_width,
                gridding_kernel_beta=gridding_kernel_beta,
            )
            if kind == "gridded" and ctx.dataset is not None:
                n_eff = getattr(ctx.dataset, "effective_n", None)
                ctx._n_eff_after_grid = float(n_eff) if n_eff is not None else None


class DefaultObjectiveFactoryStep:
    """Set default objective_factory when None (ChiSquared + L1 from lambda_l_norm, wavelet)."""

    def run(self, ctx) -> None:
        if getattr(ctx, "objective_factory", None) is None:
            lambda_l_norm = getattr(ctx, "lambda_l_norm", 0.0)
            # Scale regularization when gridded so data-term vs L1 balance matches direct case
            n_before = getattr(ctx, "_n_eff_before_grid", None)
            n_after = getattr(ctx, "_n_eff_after_grid", None)
            if (
                n_before is not None
                and n_after is not None
                and n_after > 0
                and lambda_l_norm != 0
            ):
                lambda_l_norm = lambda_l_norm * (n_before / n_after)
            ctx.objective_factory = default_objective_factory(
                lambda_l_norm,
                getattr(ctx, "wavelet", None),
            )


class DefaultOptimizerFactoryStep:
    """Set default optimizer_factory when None (FISTA)."""

    def run(self, ctx) -> None:
        if getattr(ctx, "optimizer_factory", None) is None:
            ctx.optimizer_factory = make_fista_optimizer()


class FlagDataStep:
    """Run flagger when present."""

    def run(self, ctx) -> None:
        if getattr(ctx, "flagger", None) is not None:
            ctx.flagger.run()


class DirtyMapStep:
    """Compute dirty map and initialize parameter.data to zeros (with optional wavelet decomp)."""

    def run(self, ctx) -> None:
        ctx.fd_dirty = ctx.measurement_operator.dirty_spectrum(ctx.dataset.data)
        n_phi = ctx.parameter.n
        ctx.parameter.data = np.zeros(n_phi, dtype=np.complex64)
        if getattr(ctx, "wavelet", None) is not None:
            ctx.parameter.data = ctx.wavelet.decompose_complex(ctx.parameter.data)


class DirtyStatsStep:
    """Compute dirty RM and error / quadratic-interp stats."""

    def run(self, ctx) -> None:
        dirty_noise = calculate_fd_signal_noise(
            ctx.fd_dirty,
            ctx.parameter.phi,
            ctx.parameter.max_faraday_depth,
        )
        ctx.rm_dirty = _get_rm(ctx, ctx.fd_dirty)
        ctx.rm_dirty_error = calculate_sigma_phi_peak(
            ctx.parameter.rmtf_fwhm,
            float(np.max(np.abs(ctx.fd_dirty))),
            dirty_noise,
        )
        (
            ctx.rm_dirty_quadratic_interpolation,
            ctx.dirty_peak_quadratic_interpolation,
        ) = estimate_peak_quadratic_interpolation(
            ctx.fd_dirty, ctx.parameter.cellsize
        )
        ctx.rm_dirty_quadratic_interpolation_error = calculate_sigma_phi_peak(
            ctx.parameter.rmtf_fwhm,
            ctx.dirty_peak_quadratic_interpolation,
            dirty_noise,
        )


class OptimizationStep:
    """Build objective, run optimizer, set fd_model and second_moment."""

    def run(self, ctx) -> None:
        F_obj = ctx.objective_factory(ctx.measurement_operator, ctx.parameter)
        opt = ctx.optimizer_factory(ctx.parameter, F_obj)
        obj, X = opt.run()
        ctx.coefficients = X.data
        if getattr(ctx, "wavelet", None) is not None:
            X.data = ctx.wavelet.reconstruct_complex(X.data)
        ctx.fd_model = X.data
        ctx.rm_model = _get_rm(ctx, ctx.fd_model)
        ctx.second_moment = calculate_second_moment(ctx.parameter.phi, ctx.fd_model)


class RestorationStep:
    """Compute residual and restored map (CLEAN-style)."""

    def run(self, ctx) -> None:
        ctx.fd_residual = ctx.measurement_operator.dirty_spectrum(
            ctx.dataset.data - ctx.dataset.model_data
        )
        conv_model = ctx.parameter.convolve(x=ctx.fd_model)
        pixels_per_rmtf = ctx.parameter.rmtf_fwhm / ctx.parameter.cellsize
        ctx.fd_restored = conv_model * pixels_per_rmtf + ctx.fd_residual


class RestoredStatsStep:
    """Compute restored RM and error / quadratic-interp stats; optional debug print."""

    def run(self, ctx) -> None:
        def _peak(a):
            return float(np.max(np.abs(np.asarray(asnumpy(a)))))

        def _sumabs(a):
            return float(np.sum(np.abs(np.asarray(asnumpy(a)))))

        pixels_per_rmtf = ctx.parameter.rmtf_fwhm / ctx.parameter.cellsize
        conv_model = ctx.parameter.convolve(x=ctx.fd_model)
        conv_Jy_rmtf = conv_model * pixels_per_rmtf
        print("[restore DEBUG]")
        print(
            "  cellsize=%.6f  rmtf_fwhm=%.6f  pixels_per_rmtf=%.4f"
            % (ctx.parameter.cellsize, ctx.parameter.rmtf_fwhm, pixels_per_rmtf)
        )
        print(
            "  peak:  dirty=%.6e  model=%.6e  residual=%.6e"
            % (_peak(ctx.fd_dirty), _peak(ctx.fd_model), _peak(ctx.fd_residual))
        )
        print(
            "  peak:  conv_model(Jy/phi)=%.6e  conv*ppr(Jy/rmtf)=%.6e  restored=%.6e"
            % (_peak(conv_model), _peak(conv_Jy_rmtf), _peak(ctx.fd_restored))
        )
        print(
            "  sum|model|=%.6e  sum|conv_model|=%.6e"
            % (_sumabs(ctx.fd_model), _sumabs(conv_model))
        )
        print(
            "  ratio dirty_peak/model_peak=%.4f  (expect ~pixels_per_rmtf=%.4f)"
            % (
                _peak(ctx.fd_dirty) / (_peak(ctx.fd_model) + 1e-30),
                pixels_per_rmtf,
            )
        )
        restored_noise = calculate_fd_signal_noise(
            ctx.fd_restored,
            ctx.parameter.phi,
            ctx.parameter.max_faraday_depth,
        )
        ctx.rm_restored = _get_rm(ctx, ctx.fd_restored)
        ctx.rm_restored_error = calculate_sigma_phi_peak(
            ctx.parameter.rmtf_fwhm,
            float(np.max(np.abs(ctx.fd_restored))),
            restored_noise,
        )
        (
            ctx.rm_restored_quadratic_interpolation,
            ctx.restored_peak_quadratic_interpolation,
        ) = estimate_peak_quadratic_interpolation(
            ctx.fd_restored, ctx.parameter.cellsize
        )
        ctx.rm_restored_quadratic_interpolation_error = calculate_sigma_phi_peak(
            ctx.parameter.rmtf_fwhm,
            ctx.restored_peak_quadratic_interpolation,
            restored_noise,
        )
