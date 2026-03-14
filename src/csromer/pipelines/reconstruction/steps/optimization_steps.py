"""
Optimization (CS-ROMER) pipeline steps.

Each step runs with ctx = CSROMERReconstructorWrapper instance.
"""
from __future__ import annotations

import numpy as np

from csromer.utils.array_utils import asnumpy
from csromer.utils.utilities import calculate_noise

from ..defaults import build_measurement_operator, build_parameter, default_objective_factory
from ..optimizer_factories import make_fista_optimizer
from ..reconstruction_stats import (
    calculate_fd_signal_noise,
    calculate_ricean_peak,
    calculate_second_moment,
    calculate_sigma_phi_peak,
    estimate_peak_quadratic_interpolation,
)


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
        ctx.rm_dirty = ctx.get_rm(ctx.fd_dirty)
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
        # Build objective for current parameter / operator
        objective_factory = ctx.objective_factory
        optimizer_factory = ctx.optimizer_factory

        def _run_single_lambda():
            """
            Run a single optimization with the current ctx.lambda_l_norm
            as the value for the L1 regularization term.
            """
            # The DefaultObjectiveFactoryStep has already baked lambda_l_norm into objective_factory,
            # so here we just instantiate and run the optimizer for the current configuration.
            F_obj_local = objective_factory(ctx.measurement_operator, ctx.parameter)
            # Ensure any non-differentiable term uses the provided lambda (if set on ctx).
            lam_nondiff = F_obj_local.get_lambda_nondiff()
            lambda_l_norm_ctx = getattr(ctx, "lambda_l_norm", None)
            if lam_nondiff is not None and lambda_l_norm_ctx is not None:
                F_obj_local.set_lambda_nondiff(lambda_l_norm_ctx)
            opt_local = optimizer_factory(ctx.parameter, F_obj_local)
            obj_val, X_local = opt_local.run()
            return obj_val, X_local, F_obj_local

        if getattr(ctx, "adaptive_lambda", False):
            # Outer loop: adjust L1 λ to drive Chi-squared (differentiable part) toward target_chi2.
            target_chi2 = float(getattr(ctx, "target_chi2", 1.0))
            gamma = float(getattr(ctx, "lambda_update_gamma", 0.5))
            lambda_min = float(getattr(ctx, "lambda_min", 0.0))
            lambda_max = float(getattr(ctx, "lambda_max", np.inf))
            max_updates = int(getattr(ctx, "max_lambda_updates", 5))

            # Start from ctx.lambda_l_norm (already possibly scaled for gridding).
            lam = float(getattr(ctx, "lambda_l_norm", 0.0))
            best_X = None
            best_obj = None
            best_lambda = lam
            for k in range(max_updates):
                # Use current lam as the L1 regularization strength.
                ctx.lambda_l_norm = lam
                obj_val, X_k, F_obj_k = _run_single_lambda()
                # Warm start next λ by using current solution as new initial guess.
                ctx.parameter.data = X_k.data
                # Evaluate differentiable-only part as Chi-squared surrogate
                chi2_val = F_obj_k.calculate_function(X_k.data, differentiable_only=True)
                if getattr(ctx, "verbose", True):
                    print(
                        "[adaptive-λ] step={}  lambda={:.6g}  chi2={:.6g}  target={:.6g}".format(
                            k, lam, chi2_val, target_chi2
                        )
                    )
                # If chi2 is at or below target (within relative tolerance), stop and keep this λ.
                chi2_rel_tol = float(getattr(ctx, "chi2_target_rel_tol", 0.05))
                chi2_accept = target_chi2 * (1.0 + chi2_rel_tol)
                if target_chi2 > 0.0 and chi2_val <= chi2_accept:
                    X = X_k
                    ctx.lambda_l_norm = lam
                    break

                # Otherwise, keep track of the best (closest-to-target) chi2 in case we never get below.
                if target_chi2 > 0.0:
                    cur_err = abs(chi2_val - target_chi2)
                    if best_X is None or cur_err < abs(best_obj - target_chi2):
                        best_X = X_k
                        best_obj = chi2_val
                        best_lambda = lam
                # Update λ multiplicatively (negative feedback toward target_chi2).
                # When chi2_val > target_chi2 (under-regularized), ratio < 1 and λ decreases.
                # When chi2_val < target_chi2 (over-regularized / overfitting), ratio > 1 and λ increases.
                if chi2_val > 0.0 and target_chi2 > 0.0:
                    ratio = target_chi2 / chi2_val
                    lam = lam * (ratio ** gamma)
                    lam = float(np.clip(lam, lambda_min, lambda_max))
                else:
                    # If chi2 is non-positive or target is invalid, stop updating
                    # and fall back to the best solution seen so far (if any).
                    if best_X is not None:
                        X = best_X
                        ctx.lambda_l_norm = best_lambda
                    break
            else:
                # If loop ended without break, fall back to best seen solution (by chi2 closeness).
                if best_X is not None:
                    X = best_X
                    ctx.lambda_l_norm = best_lambda
        else:
            # Standard single-run optimization with fixed λ.
            obj, X, _ = _run_single_lambda()

        ctx.coefficients = X.data
        if getattr(ctx, "wavelet", None) is not None:
            X.data = ctx.wavelet.reconstruct_complex(X.data)
        ctx.fd_model = X.data

        ctx.rm_model = ctx.get_rm(ctx.fd_model)
        ctx.second_moment = calculate_second_moment(ctx.parameter.phi, ctx.fd_model)


class RestorationStep:
    """Compute residual and restored map (CLEAN-style)."""

    def run(self, ctx) -> None:
        ctx.fd_residual = ctx.measurement_operator.dirty_spectrum(
            ctx.dataset.data - ctx.dataset.model_data
        )
        # Convolve complex Faraday spectrum and its amplitude with the clean beam.
        conv_model, conv_abs_model = ctx.parameter.convolve(x=ctx.fd_model)
        # Cache for later reuse in RestoredStatsStep.
        ctx.conv_model = conv_model
        ctx.conv_abs_model = conv_abs_model
        # Restored complex spectrum: CLEAN-style model + residual (same units as dirty).
        ctx.fd_restored = conv_model + ctx.fd_residual
        # Restored amplitude: |model| restored + |residual|.
        ctx.fd_restored_abs = conv_abs_model + np.abs(ctx.fd_residual)


class RestoredStatsStep:
    """Compute restored RM and error / quadratic-interp stats; optional debug print."""

    def run(self, ctx) -> None:
        def _peak(a):
            return float(np.max(np.abs(np.asarray(asnumpy(a)))))

        def _sumabs(a):
            return float(np.sum(np.abs(np.asarray(asnumpy(a)))))

        pixels_per_rmtf = getattr(
            ctx,
            "pixels_per_rmtf",
            ctx.parameter.rmtf_fwhm / ctx.parameter.cellsize,
        )
        # Complex model convolved with clean beam (from RestorationStep).
        conv_model = getattr(ctx, "conv_model", ctx.parameter.convolve(x=ctx.fd_model)[0])
        # Amplitude |F| convolved with clean beam (if available).
        conv_abs_model = getattr(
            ctx,
            "conv_abs_model",
            ctx.parameter.convolve(x=ctx.fd_model)[1],
        )

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
            "  peak:  conv_model(Jy/rmtf)=%.6e restored=%.6e"
            % (_peak(conv_model), _peak(ctx.fd_restored))
        )
        print(
            "  sum|model|=%.6e  sum|conv_model|=%.6e"
            % (_sumabs(ctx.fd_model), _sumabs(conv_model))
        )
        # Optional diagnostic for amplitude-restored spectrum.
        print(
            "  peak:  conv_abs_model=%.6e (amp-restored)"
            % (_peak(conv_abs_model),)
        )
        print(
            "  ratio dirty_peak/model_peak=%.4f  (expect ~pixels_per_rmtf=%.4f)"
            % (
                _peak(ctx.fd_dirty) / (_peak(ctx.fd_model) + 1e-30),
                pixels_per_rmtf,
            )
        )
        # Channel-space (lambda^2) residual vs thermal noise
        dataset_residual = getattr(ctx.dataset, "residual", None)
        dataset_noise = getattr(ctx.dataset, "noise", None)
        if dataset_residual is not None and dataset_noise is not None:
            res_vis = np.asarray(asnumpy(dataset_residual))
            rms_vis = float(np.sqrt(np.mean(np.abs(res_vis) ** 2)))
            print(
                "  vis-space: rms(residual)=%.6e  noise=%.6e  ratio=%.4f"
                % (rms_vis, float(dataset_noise), rms_vis / (float(dataset_noise) + 1e-30))
            )
        # Faraday-depth residual noise level (MAD-based, robust to correlation/outliers).
        fd_res = np.asarray(asnumpy(ctx.fd_residual))
        # calculate_noise expects at least a 2D image (y, x). For a 1D Faraday spectrum,
        # reshape to (n_phi, 1) so that indexing image[y0:yn, x0:xn] is valid.
        fd_res_img = fd_res[:, np.newaxis]
        mad_fd_res = float(
            calculate_noise(
                image=fd_res_img,
                use_sigma_clipped_stats=True,
                stdfunc="mad_std",
            )
        )
        # Use the MAD-based Faraday-depth noise from the residual spectrum as the
        # reference noise level for RM error estimation. This ties the RM error
        # directly to the fd-space noise (mad_std) instead of re-estimating it
        # from the restored spectrum, and avoids unrealistically small errors.
        restored_noise = mad_fd_res
        # Diagnostics: compare FD-space noise estimate, theoretical channel noise,
        # and RMS levels of residual/restored spectra.
        dataset_theo_noise = getattr(ctx.dataset, "theo_noise", None)
        if dataset_theo_noise is not None:
            print("  theo_noise (chan)=%.6e" % float(dataset_theo_noise))
        fd_res_amp = np.abs(fd_res)
        fd_rest_amp = np.abs(np.asarray(asnumpy(ctx.fd_restored)))
        rms_fd_res = float(np.sqrt(np.mean(fd_res_amp**2)))
        rms_fd_rest = float(np.sqrt(np.mean(fd_rest_amp**2)))
        print(
            "  fd-space: mad_std(fd_residual)=%.6e  rms(|fd_residual|)=%.6e  rms(|fd_restored|)=%.6e"
            % (mad_fd_res, rms_fd_res, rms_fd_rest)
        )
        # RM at the peak from the model (grid-based; kept for diagnostics).
        ctx.rm_peak = ctx.get_rm(ctx.fd_model)
        # Peak amplitude from fd_restored_abs at model peak, Rician-corrected; use residual noise for error.
        fd_restored_abs = np.asarray(asnumpy(ctx.fd_restored_abs))
        peak_idx = int(np.argmax(np.abs(np.asarray(asnumpy(ctx.fd_model)))))
        peak_restored_abs = float(fd_restored_abs[peak_idx])
        peak_restored_abs_corrected = calculate_ricean_peak(peak_restored_abs, restored_noise)
        (
            ctx.rm_restored_quadratic_interpolation,
            ctx.restored_peak_quadratic_interpolation,
        ) = estimate_peak_quadratic_interpolation(
            ctx.fd_restored, ctx.parameter.cellsize
        )
        # Use the Ricean-corrected peak and residual noise for the quadratic-interp RM error,
        # and adopt the quadratic-interpolated RM and its error as the canonical restored values.
        ctx.rm_restored_quadratic_interpolation_error = calculate_sigma_phi_peak(
            ctx.parameter.rmtf_fwhm,
            peak_restored_abs_corrected,
            restored_noise,
        )
        ctx.rm_restored = ctx.rm_restored_quadratic_interpolation
        ctx.rm_restored_error = ctx.rm_restored_quadratic_interpolation_error
