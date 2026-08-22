"""
Regression tests for two behaviors observed by inspection of the SKAO book notebooks
(see examples/04_reconstruction_cg.ipynb, PolakRibiere with lambda_l_norm=0.8). These
looked at first like a single "L1 is non-differentiable" story, but empirically they
are two *distinct* problems -- see the investigation notes below for how that was
disentangled.

1. Non-linear CG variants could *increase* the objective function value across
   iterations instead of monotonically decreasing it, once L1 regularization was
   active. Root cause (confirmed by tracing per-iteration alpha in
   ConjugateGradient._line_search): L1.calculate_gradient's smoothing
   epsilon (src/csromer/objectivefunction/priors/l1.py) defaulted to
   np.finfo(np.float32).tiny (~1.18e-38) -- not a real smoothing scale, just a
   division-by-zero guard, so sqrt(x**2 + epsilon) is bit-identical to |x| in
   float64 for any x above ~1e-19. The objective therefore has a genuine
   floating-point kink, and Armijo backtracking can legitimately exhaust its
   `max_ls` budget without ever satisfying the sufficient-decrease condition near
   one. The *real* bug this exposed was in `_line_search` itself: on exhausting
   `max_ls`, it returned the smallest trial alpha regardless of whether it actually
   improved on f_x -- so a non-improving (or worsening) step got taken every single
   time this happened. Traced directly: for PolakRibiere+L1 on a small test problem,
   alpha collapsed to exactly `1.0 * rho**30` on *every* iteration from iteration 6
   onward -- a full stall, not an occasional glitch, and consistent with the slow
   multi-thousand-iteration upward drift seen in the notebook's saved output
   (~10.19392 at iteration 11 climbing to ~10.20069 by iteration 4991).
   `_line_search` now tracks the best (lowest-f) trial point seen during
   backtracking and only ever returns a step that does not increase f (0.0 if none
   does), which fixes this at the source -- see
   test_cg_monotonic_decrease_never_increases_objective below, which now holds
   unconditionally for both smooth and L1-regularized objectives.

2. The reconstructed (model) Faraday-depth peak amplitude falls well short of the
   dirty-spectrum peak. RestoredStatsStep (src/csromer/pipelines/reconstruction/
   steps/optimization_steps.py) already prints a "ratio dirty_peak/model_peak" vs.
   "expect ~pixels_per_rmtf" diagnostic for this; in the notebook run this ratio was
   24.66 against an expected ~6.92. Initially this looked like it might share the
   same root cause as (1), but that was ruled out experimentally: sweeping the L1
   smoothing epsilon from 1e-38 to 1e-2 (a range that does eliminate the line-search
   stall from (1)) left the recovered peak completely unchanged. Likewise sweeping
   lambda_l_norm from 0 to 0.5 barely moved it. What *did* move it: swapping in
   FISTA (proximal gradient with L1's exact soft-threshold prox) at a well-tuned
   lambda_l_norm got much closer to the pixels_per_rmtf expectation. So this is a
   distinct, known limitation of gradient/smoothing-based L1 handling vs. proximal
   splitting for inducing real sparsity -- consistent with the optimization
   literature: dedicated "smoothed conjugate gradient for L1" methods are their own
   research topic precisely because naive smoothed-gradient CG underperforms
   proximal methods here (e.g. "A new smoothing modified three-term conjugate
   gradient method for l1-norm minimization problem", PMC5934501). Not fixed here;
   tracked as a regression guard in test_model_peak_recovers_reasonable_fraction_of_dirty_peak.

Marked `slow` because they run several full reconstructions per CG variant; run
explicitly with:
    pytest -m slow -v tests/integration/test_optimizer_monotonicity.py
"""
import numpy as np
import pytest

pytest.importorskip("pywt", reason="requires PyWavelets (pip install PyWavelets)")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from csromer.optimization import DaiYuan, FletcherReeves, HagerZhang, HestenesStiefel, PolakRibiere
from csromer.pipelines.reconstruction import CSROMERReconstructorWrapper, make_cg_optimizer
from csromer.simulation import FaradayThinSource

CG_METHODS = [PolakRibiere, FletcherReeves, HestenesStiefel, DaiYuan, HagerZhang]

# Relative tolerance for "did the objective increase" (guards against float noise).
_REL_TOL = 1e-9


def _tracking_cg_factory(method, maxiter, tol):
    """
    Wrap make_cg_optimizer so the accepted objective value at the end of every CG
    iteration is recorded. ConjugateGradient.run() only returns the final cost, so
    per-iteration history has to be captured by hooking _perform_iteration.
    """
    history = []
    base_factory = make_cg_optimizer(method=method, maxiter=maxiter, tol=tol, verbose=False)

    def factory(parameter, F_obj):
        opt = base_factory(parameter, F_obj)
        original = opt._perform_iteration

        def tracked(iteration, current_param, prev_gradient, prev_search_direction):
            result = original(iteration, current_param, prev_gradient, prev_search_direction)
            history.append(result[1])  # new_function_value
            return result

        opt._perform_iteration = tracked
        return opt

    return factory, history


def _make_source():
    nu = np.linspace(1.0e9, 1.5e9, 64)
    source = FaradayThinSource(nu=nu, s_nu=0.1, phi_gal=50.0, spectral_idx=-0.7)
    source.simulate()
    source.apply_noise(0.01, random_state=np.random.RandomState(42))
    return source


def _increases(history):
    return [
        (i, prev, cur) for i, (prev, cur) in enumerate(zip(history, history[1:]))
        if cur > prev + _REL_TOL * max(abs(prev), 1.0)
    ]


@pytest.mark.parametrize("lambda_l_norm", [0.0, 0.5], ids=["smooth", "l1_regularized"])
@pytest.mark.parametrize("cg_method", CG_METHODS)
def test_cg_monotonic_decrease_never_increases_objective(cg_method, lambda_l_norm):
    """
    ConjugateGradient._line_search now only ever returns a step that does not
    increase f (falling back to 0.0 -- no step -- if nothing tried during
    backtracking improves on f_x). This should hold unconditionally: both for a
    smooth objective (no L1 term) and for an L1-regularized one, where it used to
    fail (see module docstring for how that was tracked down to the line search's
    exhausted-backtracking fallback, not to L1's non-differentiability per se).
    """
    source = _make_source()
    factory, history = _tracking_cg_factory(cg_method, maxiter=100, tol=1e-10)

    recon = CSROMERReconstructorWrapper(
        dataset=source,
        oversampling=4.0,
        optimizer_factory=factory,
        lambda_l_norm=lambda_l_norm,
        measurement_operator_kind="gridded",
    )
    recon.reconstruct()

    assert len(history) > 1, "optimizer ran too few iterations to check monotonicity"
    bad = _increases(history)
    assert not bad, (
        f"{cg_method.__name__} (lambda_l_norm={lambda_l_norm}): objective increased "
        f"at iterations {[i for i, _, _ in bad][:5]} "
        f"(showing up to 5 of {len(bad)}): {bad[:5]}"
    )


@pytest.mark.parametrize("cg_method", CG_METHODS)
def test_model_peak_recovers_reasonable_fraction_of_dirty_peak(cg_method):
    """
    Regression guard for the flux-loss behavior visible in
    examples/04_reconstruction_cg.ipynb's "[restore DEBUG]" print: for a single
    point-like (thin) source, RestoredStatsStep already computes and prints
        ratio dirty_peak/model_peak   vs.   expect ~pixels_per_rmtf = rmtf_fwhm/cellsize
    In the notebook run this ratio was 24.66 against an expected ~6.92 (model
    recovers ~3.5x less flux than expected). This is NOT the same root cause as the
    line-search monotonicity issue above (confirmed by sweeping both the L1
    smoothing epsilon and lambda_l_norm -- neither moves this ratio); it reflects
    CG's smoothed-gradient handling of L1 being weaker at inducing sparsity than a
    proximal method like FISTA (see module docstring). This test recomputes the
    same ratio for a small synthetic source and asserts it stays within a generous
    band around pixels_per_rmtf, so severe additional flux loss gets caught even
    though the exact numeric relationship is not (yet) derived from first
    principles here.
    """
    source = _make_source()

    recon = CSROMERReconstructorWrapper(
        dataset=source,
        oversampling=4.0,
        optimizer_factory=make_cg_optimizer(
            method=cg_method, maxiter=100, tol=1e-10, verbose=False
        ),
        lambda_l_norm=0.5,
        measurement_operator_kind="gridded",
    )
    recon.reconstruct()

    dirty_peak = float(np.max(np.abs(np.asarray(recon.fd_dirty))))
    model_peak = float(np.max(np.abs(np.asarray(recon.fd_model))))
    assert model_peak > 0.0, f"{cg_method.__name__}: model peak is zero"

    pixels_per_rmtf = recon.parameter.rmtf_fwhm / recon.parameter.cellsize
    ratio = dirty_peak / model_peak

    # Generous band (0.25x-2x of pixels_per_rmtf): tight enough to catch the ~3.5x
    # excess loss seen in the notebook, loose enough not to be a source of flakiness
    # across CG variants/seeds.
    assert 0.25 * pixels_per_rmtf <= ratio <= 2.0 * pixels_per_rmtf, (
        f"{cg_method.__name__}: dirty_peak/model_peak ratio={ratio:.3f} is far from "
        f"pixels_per_rmtf={pixels_per_rmtf:.3f} (dirty_peak={dirty_peak:.4e}, "
        f"model_peak={model_peak:.4e}) -- model is recovering much less (or more) "
        f"flux than expected"
    )
