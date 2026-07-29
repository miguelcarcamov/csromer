"""
Amplitude / flux-recovery audit for the Faraday reconstruction pipeline.

Uses the public API (FaradayThinSource, DirectFourier1D, CLEAN/CSROMER wrappers)
to check whether a known synthetic point source recovers P0, phi0, and chi0
through dirty synthesis and through RMCLEAN / RML + restoration.

Hypothesis coverage:
  H1 restoring-beam peak vs integral normalization
  H2 lambda_0^2 consistency (forward vs dirty/residual)
  H3 FFT / adjoint normalization
  H4 gridding / RMTF K = 1/sum(w)

See docs/amplitude_audit_report.md for the written verdict.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt", reason="reconstruction wrappers import wavelet stack")

from csromer.pipelines.reconstruction import (
    CLEANReconstructorWrapper,
    CSROMERReconstructorWrapper,
    make_cg_optimizer,
)
from csromer.reconstruction import Parameter
from csromer.simulation import FaradayThinSource
from csromer.simulation.bands import SKA_MID_B2
from csromer.transformers.measurement_operator import DirectFourier1D
from csromer.utils.array_utils import asnumpy

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window_peak(fd, phi, phi0, half_width):
    """Peak |F|, phi, chi=0.5*arg(F) inside |phi-phi0| < half_width."""
    fd = np.asarray(asnumpy(fd))
    phi = np.asarray(asnumpy(phi))
    mask = np.abs(phi - phi0) < half_width
    assert np.any(mask), f"No phi samples within {half_width} of {phi0}"
    idx_local = int(np.argmax(np.abs(fd[mask])))
    idx = np.where(mask)[0][idx_local]
    val = fd[idx]
    return {
        "P": float(np.abs(val)),
        "phi": float(phi[idx]),
        "chi": 0.5 * float(np.angle(val)),
        "idx": int(idx),
    }


def _inject_thin_source(nu, P0, phi0, chi0=0.0, l2_ref=0.0, remove_frac=0.0, seed=42):
    """
    Build a noiseless thin source with correct complex phase.

    FaradayThinSource.simulate() applies dchi only to U (not Q); for chi tests we
    overwrite data with P = P0 * exp(2j*(chi0 + phi0*lambda^2)).
    """
    src = FaradayThinSource(nu=nu, s_nu=P0, phi_gal=phi0, spectral_idx=0.0, dchi=0.0)
    src.l2_ref = float(l2_ref)
    src.simulate()
    l2 = np.asarray(asnumpy(src.lambda2))
    src.data = (P0 * np.exp(2.0j * (chi0 + phi0 * l2))).astype(np.complex64)
    if remove_frac > 0.0:
        src.remove_channels(remove_frac=remove_frac, random_state=np.random.RandomState(seed))
    return src


def _ska_mid_b2_subsampled(n_target=128):
    nu = SKA_MID_B2.freq_array()
    step = max(1, len(nu) // n_target)
    return nu[::step]


# ---------------------------------------------------------------------------
# H1 — restoring beam is peak-normalized (max=1)
# ---------------------------------------------------------------------------


def test_h1_clean_beam_is_peak_normalized():
    """Gaussian restore beam must have peak=1 (flux-preserving for delta components)."""
    nu = np.linspace(1.0e9, 1.5e9, 64)
    src = _inject_thin_source(nu, P0=1.0, phi0=0.0)
    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=4.0, verbose=False)
    kernel = param._clean_beam_kernel(param.rmtf_fwhm)
    np.testing.assert_allclose(float(np.max(kernel)), 1.0, rtol=1e-6)
    # Sum-normalized would give max ≈ 1/(sigma*sqrt(2*pi)) << 1 for resolved beams
    assert float(np.sum(kernel)) > 1.0 + 1e-3, (
        "Peak-normalized Gaussian should have sum > 1; sum≈1 suggests integral normalization"
    )


def test_h1_delta_convolve_preserves_peak():
    """Delta of amplitude 1 convolved with clean beam → peak 1."""
    nu = np.linspace(1.0e9, 1.5e9, 64)
    src = _inject_thin_source(nu, P0=1.0, phi0=0.0)
    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=4.0, verbose=False)
    fd = np.zeros(param.n, dtype=np.complex64)
    fd[param.n // 2] = 1.0 + 0.0j
    complex_restored, _ = param.convolve(x=fd, rmtf_fwhm=param.rmtf_fwhm)
    peak = float(np.max(np.abs(np.asarray(asnumpy(complex_restored)))))
    np.testing.assert_allclose(peak, 1.0, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# H4 — RMTF / dirty normalization
# ---------------------------------------------------------------------------


def test_h4_rmtf_peak_is_one():
    """K = 1/sum(w) so RMTF(phi=0) peak = 1."""
    nu = np.linspace(1.0e9, 1.5e9, 128)
    src = _inject_thin_source(nu, P0=1.0, phi0=0.0)
    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=4.0, verbose=False)
    op = DirectFourier1D(dataset=src, parameter=param)
    rmtf = np.asarray(asnumpy(op.RMTF()))
    np.testing.assert_allclose(float(np.max(np.abs(rmtf))), 1.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Matched complete coverage — model peak must equal P0 (restoration irrelevant)
# ---------------------------------------------------------------------------


def test_matched_complete_rml_model_peak_equals_p0():
    """
    Noiseless point source on a Nyquist-matched grid (n_phi == n_chan, uniform λ²).

    This is the well-posed case: RML (CG, λ=0) must recover model peak ≈ P0
    *without relying on restoration*. Restored peak must also be ≈ P0.
    """
    from csromer.base import Dataset

    P0 = 1.0
    n = 128
    l2 = np.linspace(0.02, 0.12, n)
    d_l2 = float(l2[1] - l2[0])
    d_phi = np.pi / (n * d_l2)
    phi = d_phi * (np.arange(n) - n // 2)

    param = Parameter(phi=phi, cellsize=d_phi, data=np.zeros(n, dtype=np.complex64))
    param.rmtf_fwhm = 2.0 * np.sqrt(3.0) / (l2.max() - l2.min())
    param.max_faraday_depth = float(np.max(np.abs(phi)))
    param.max_recovered_width = float(np.pi / l2.min())

    F_true = np.zeros(n, dtype=np.complex64)
    F_true[n // 2] = P0

    ds = Dataset(lambda2=l2, w=np.ones(n, dtype=np.float64))
    ds.l2_ref = 0.0
    op = DirectFourier1D(dataset=ds, parameter=param)
    ds.data = np.asarray(op.forward(F_true), dtype=np.complex64)

    recon = CSROMERReconstructorWrapper(
        dataset=ds,
        parameter=param,
        measurement_operator=DirectFourier1D(dataset=ds, parameter=param),
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=200, tol=1e-10, verbose=False),
    )
    recon.reconstruct()

    model_peak = float(np.max(np.abs(np.asarray(asnumpy(recon.fd_model)))))
    rest_peak = float(np.max(np.abs(np.asarray(asnumpy(recon.fd_restored)))))
    np.testing.assert_allclose(model_peak, P0, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(rest_peak, P0, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(asnumpy(recon.fd_model)), F_true, atol=1e-5, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Test A — dense coverage, dirty only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi0", [-80.0, 0.0, 50.0])
@pytest.mark.parametrize("chi0", [0.0, 0.25])
def test_a_dense_dirty_recovers_p0_phi_chi(phi0, chi0):
    """
    Dense frequency sampling, dirty FDF only (no deconvolution).
    Peak |F_dirty| ≈ P0, phi ≈ phi0, chi ≈ chi0.
    """
    P0 = 1.0
    nu = np.linspace(1.0e9, 2.0e9, 512)
    src = _inject_thin_source(nu, P0=P0, phi0=phi0, chi0=chi0, l2_ref=0.0)
    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=8.0, verbose=False)
    op = DirectFourier1D(dataset=src, parameter=param)
    dirty = op.dirty_spectrum(src.data)
    half = max(2.0 * param.rmtf_fwhm, 3.0 * param.cellsize)
    pk = _window_peak(dirty, param.phi, phi0, half)

    assert abs(pk["P"] - P0) / P0 < 0.02, (
        f"Test A amplitude: recovered {pk['P']:.4f}, expected {P0}"
    )
    assert abs(pk["phi"] - phi0) < max(param.cellsize * 2.0, param.rmtf_fwhm * 0.5), (
        f"Test A phi: recovered {pk['phi']:.3f}, expected {phi0}"
    )
    # Phase: allow wrapping and small discrete-bin error
    dchi = (pk["chi"] - chi0 + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dchi) < 0.15, (
        f"Test A chi: recovered {pk['chi']:.4f}, expected {chi0} (dchi={dchi:.4f})"
    )


# ---------------------------------------------------------------------------
# Test B — realistic incomplete coverage + restoration
# ---------------------------------------------------------------------------


@pytest.fixture
def realistic_thin_source():
    """SKA-MID B2 subsampled + 25% channel gaps, 1 Jy at phi=50, chi=0, l2_ref=0."""
    return _inject_thin_source(
        _ska_mid_b2_subsampled(128),
        P0=1.0,
        phi0=50.0,
        chi0=0.0,
        l2_ref=0.0,
        remove_frac=0.25,
        seed=42,
    )


def test_b_dirty_recovers_p0_before_deconv(realistic_thin_source):
    """Even with gaps, dirty peak of a 1 Jy point source must be ≈ 1 (RMSF peak=1)."""
    src = realistic_thin_source
    P0, phi0 = 1.0, 50.0
    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=6.0, verbose=False)
    op = DirectFourier1D(dataset=src, parameter=param)
    dirty = op.dirty_spectrum(src.data)
    pk = _window_peak(dirty, param.phi, phi0, 3.0 * param.rmtf_fwhm)
    assert abs(pk["P"] - P0) / P0 < 0.03
    assert abs(pk["phi"] - phi0) < param.rmtf_fwhm


def test_b_rmclean_restored_recovers_p0(realistic_thin_source):
    """
    RMCLEAN end-to-end including restoration: peak |F_restored| near phi0 ≈ P0.

    Uses a window around phi0 because periodic RMTF shifting can deposit
    spurious edge components (separate issue; see report).
    """
    src = realistic_thin_source
    P0, phi0, chi0 = 1.0, 50.0, 0.0
    recon = CLEANReconstructorWrapper(
        dataset=src,
        oversampling=6.0,
        clean_maxiter=400,
        clean_gain=0.1,
        clean_threshold=1e-3,
    )
    recon.reconstruct()
    half = 3.0 * recon.parameter.rmtf_fwhm
    pk = _window_peak(recon.fd_restored, recon.parameter.phi, phi0, half)
    assert abs(pk["phi"] - phi0) < recon.parameter.rmtf_fwhm
    dchi = (pk["chi"] - chi0 + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dchi) < 0.2
    assert abs(pk["P"] - P0) / P0 < 0.08, (
        f"CLEAN restored peak {pk['P']:.4f} vs P0={P0} (window around phi0)"
    )


def test_b_rml_near_zero_reg_restored_recovers_p0(realistic_thin_source):
    """
    RML (CG, lambda_l_norm=0) end-to-end including CLEAN-style restoration.

    Expected (correct behaviour): restored peak ≈ P0.
    Current implementation attenuates strongly because RestorationStep convolves a
    spread non-sparse model with a peak-normalized Gaussian (see report).
    """
    src = realistic_thin_source
    P0, phi0, chi0 = 1.0, 50.0, 0.0
    recon = CSROMERReconstructorWrapper(
        dataset=src,
        oversampling=6.0,
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=80, tol=1e-6, verbose=False),
    )
    recon.reconstruct()
    half = 3.0 * recon.parameter.rmtf_fwhm
    pk = _window_peak(recon.fd_restored, recon.parameter.phi, phi0, half)

    # Position / phase should still be OK
    assert abs(pk["phi"] - phi0) < recon.parameter.rmtf_fwhm
    dchi = (pk["chi"] - chi0 + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dchi) < 0.25

    # Diagnostic: model fits data — dirty(A F) recovers P0 even if restored does not
    op = recon.measurement_operator
    model_dirty = np.asarray(asnumpy(op.dirty_spectrum(op.forward(recon.fd_model))))
    md_pk = _window_peak(model_dirty, recon.parameter.phi, phi0, half)
    assert abs(md_pk["P"] - P0) / P0 < 0.05, (
        f"RML model fits data (dirty(AF)={md_pk['P']:.4f}) but this is a diagnostic only"
    )

    assert abs(pk["P"] - P0) / P0 < 0.15, (
        f"RML restored peak {pk['P']:.4f} vs P0={P0}: attenuation factor "
        f"{pk['P']/P0:.3f}. Isolates restoration of spread model "
        f"(model peak={float(np.max(np.abs(recon.fd_model))):.4f}, "
        f"sum|model|={float(np.sum(np.abs(recon.fd_model))):.3f})."
    )


def test_b_rml_model_dirty_matches_data_dirty(realistic_thin_source):
    """
    With lambda→0, A F ≈ data so dirty(A F) ≈ dirty(data). Amplitude loss is not in
    the measurement operator / FFT path.
    """
    src = realistic_thin_source
    recon = CSROMERReconstructorWrapper(
        dataset=src,
        oversampling=6.0,
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=80, tol=1e-6, verbose=False),
    )
    recon.reconstruct()
    op = recon.measurement_operator
    dirty = np.asarray(asnumpy(recon.fd_dirty))
    model_dirty = np.asarray(asnumpy(op.dirty_spectrum(op.forward(recon.fd_model))))
    np.testing.assert_allclose(model_dirty, dirty, atol=5e-3, rtol=0.0)


# ---------------------------------------------------------------------------
# Test C — multiple point sources; bias constant vs variable
# ---------------------------------------------------------------------------


def test_c_multi_source_rmclean_amplitude_bias():
    """
    Several thin sources at different P0/phi0 through RMCLEAN.
    Records recovered/expected ratios; asserts each window peak within tolerance.
    """
    nu = _ska_mid_b2_subsampled(96)
    sources = [
        (0.5, -40.0, 0.0),
        (1.0, 20.0, 0.0),
        (1.5, 90.0, 0.0),
    ]
    # Build combined data
    base = _inject_thin_source(nu, P0=sources[0][0], phi0=sources[0][1], chi0=0.0, l2_ref=0.0)
    data = np.zeros_like(base.data)
    for P0, phi0, chi0 in sources:
        s = _inject_thin_source(nu, P0=P0, phi0=phi0, chi0=chi0, l2_ref=0.0)
        data = data + s.data
    base.data = data.astype(np.complex64)
    base.remove_channels(remove_frac=0.2, random_state=np.random.RandomState(7))

    recon = CLEANReconstructorWrapper(
        dataset=base,
        oversampling=5.0,
        clean_maxiter=600,
        clean_gain=0.1,
        clean_threshold=2e-3,
    )
    recon.reconstruct()
    ratios = []
    for P0, phi0, chi0 in sources:
        pk = _window_peak(
            recon.fd_restored, recon.parameter.phi, phi0, 3.0 * recon.parameter.rmtf_fwhm
        )
        ratios.append(pk["P"] / P0)
        assert abs(pk["phi"] - phi0) < 2.0 * recon.parameter.rmtf_fwhm
        assert abs(pk["P"] - P0) / P0 < 0.20, (
            f"Source P0={P0} phi0={phi0}: recovered {pk['P']:.4f}, ratios so far {ratios}"
        )
    # Constant multiplicative bias → low scatter in ratios
    assert np.std(ratios) < 0.15 or np.allclose(ratios, 1.0, atol=0.2)


def test_c_multi_source_rml_attenuation_factor():
    """
    Same multi-source setup through RML (CG, λ=0).
    Documents whether attenuation is roughly constant (normalization bug) or variable.
    """
    nu = _ska_mid_b2_subsampled(96)
    sources = [
        (0.5, -40.0, 0.0),
        (1.0, 20.0, 0.0),
        (1.5, 90.0, 0.0),
    ]
    base = _inject_thin_source(nu, P0=sources[0][0], phi0=sources[0][1], chi0=0.0, l2_ref=0.0)
    data = np.zeros_like(base.data)
    for P0, phi0, chi0 in sources:
        s = _inject_thin_source(nu, P0=P0, phi0=phi0, chi0=chi0, l2_ref=0.0)
        data = data + s.data
    base.data = data.astype(np.complex64)
    base.remove_channels(remove_frac=0.2, random_state=np.random.RandomState(7))

    recon = CSROMERReconstructorWrapper(
        dataset=base,
        oversampling=5.0,
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=60, tol=1e-5, verbose=False),
    )
    recon.reconstruct()
    ratios = []
    for P0, phi0, _chi0 in sources:
        pk = _window_peak(
            recon.fd_restored, recon.parameter.phi, phi0, 3.0 * recon.parameter.rmtf_fwhm
        )
        ratios.append(pk["P"] / P0)
    # Fail with diagnostic if systematically attenuated
    mean_ratio = float(np.mean(ratios))
    assert mean_ratio > 0.7, (
        f"RML multi-source restored/P0 ratios={ratios} mean={mean_ratio:.3f} "
        f"(constant factor suggests shared restoration/normalization issue)"
    )


# ---------------------------------------------------------------------------
# Test D — RMCLEAN vs RML on the same scenario
# ---------------------------------------------------------------------------


def test_d_rmclean_vs_rml_same_scenario(realistic_thin_source):
    """
    Same synthetic data through CLEAN and RML (λ=0).
    CLEAN should recover ~P0; if only RML fails, bug is restoration-of-spread-model
    (shared beam kernel is fine); if both fail similarly, shared path is implicated.
    """
    P0, phi0 = 1.0, 50.0

    src_c = realistic_thin_source
    # Independent copy for RML (wrappers mutate dataset)
    src_r = _inject_thin_source(
        _ska_mid_b2_subsampled(128),
        P0=P0,
        phi0=phi0,
        chi0=0.0,
        l2_ref=0.0,
        remove_frac=0.25,
        seed=42,
    )

    clean = CLEANReconstructorWrapper(
        dataset=src_c,
        oversampling=6.0,
        clean_maxiter=400,
        clean_gain=0.1,
        clean_threshold=1e-3,
    )
    clean.reconstruct()
    half = 3.0 * clean.parameter.rmtf_fwhm
    pk_c = _window_peak(clean.fd_restored, clean.parameter.phi, phi0, half)

    rml = CSROMERReconstructorWrapper(
        dataset=src_r,
        oversampling=6.0,
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=80, tol=1e-6, verbose=False),
    )
    rml.reconstruct()
    half_r = 3.0 * rml.parameter.rmtf_fwhm
    pk_r = _window_peak(rml.fd_restored, rml.parameter.phi, phi0, half_r)

    clean_ok = abs(pk_c["P"] - P0) / P0 < 0.1
    rml_ok = abs(pk_r["P"] - P0) / P0 < 0.15
    assert clean_ok, f"CLEAN restored {pk_c['P']:.4f} (expected ~{P0})"
    assert rml_ok, (
        f"RML restored {pk_r['P']:.4f} (expected ~{P0}); CLEAN was {pk_c['P']:.4f}. "
        f"CLEAN-only OK ⇒ RML-specific (spread model + RestorationStep), not shared beam max≠1."
    )


# ---------------------------------------------------------------------------
# H2 — lambda_0^2: phase ramp on dirty vs un-phased model in restoration
# ---------------------------------------------------------------------------


def test_h2_l2_ref_nonzero_rml_complex_restore_cancels():
    """
    With l2_ref = <lambda^2>, dirty_spectrum applies exp(+2j phi l2_ref) but the
    measurement operator / optimizer do not. Complex Gaussian restoration of the
    (non-derotated) RML model then suffers phase cancellation across the beam:
    |conv(F)| << conv(|F|).

    This test documents the inconsistency; it asserts that complex-restored peak
    stays within a factor of ~2 of abs-restored peak (will fail if cancellation is severe).
    """
    nu = np.linspace(1.0e9, 1.5e9, 64)
    P0, phi0 = 1.0, 50.0
    src = _inject_thin_source(nu, P0=P0, phi0=phi0, chi0=0.0)
    src.l2_ref = float(np.average(np.asarray(src.lambda2), weights=np.asarray(src.w)))

    recon = CSROMERReconstructorWrapper(
        dataset=src,
        oversampling=4.0,
        lambda_l_norm=0.0,
        wavelet=None,
        optimizer_factory=make_cg_optimizer(maxiter=40, tol=1e-5, verbose=False),
    )
    recon.reconstruct()
    half = 3.0 * recon.parameter.rmtf_fwhm
    pk_c = _window_peak(recon.fd_restored, recon.parameter.phi, phi0, half)
    pk_a = _window_peak(recon.fd_restored_abs, recon.parameter.phi, phi0, half)

    # If complex restore cancelled, pk_c << pk_a
    assert pk_c["P"] > 0.5 * pk_a["P"], (
        f"l2_ref={src.l2_ref:.4e}: complex restored peak {pk_c['P']:.4f} << "
        f"abs-restored {pk_a['P']:.4f} (phase cancellation from H2 inconsistency)"
    )


def test_h2_chi_scales_with_phi0_when_l2_ref_applied_to_dirty_only():
    """
    Dirty-only: with l2_ref>0 the post-adjoint phase ramp rotates chi by ~phi*l2_ref.
    At the recovered peak, chi_meas - chi0 should track phi0 * l2_ref (mod pi/2 care).
    Forward operator still has no l2_ref — so this is a display/convention effect on dirty.
    """
    nu = np.linspace(1.0e9, 1.5e9, 128)
    phi0 = 50.0
    chi0 = 0.0
    src = _inject_thin_source(nu, P0=1.0, phi0=phi0, chi0=chi0, l2_ref=0.0)
    l2_ref = float(np.average(np.asarray(src.lambda2), weights=np.asarray(src.w)))
    src.l2_ref = l2_ref

    param = Parameter()
    param.calculate_cellsize(dataset=src, oversampling=4.0, verbose=False)
    op = DirectFourier1D(dataset=src, parameter=param)
    dirty = op.dirty_spectrum(src.data)
    pk = _window_peak(dirty, param.phi, phi0, 3.0 * param.rmtf_fwhm)

    # Expected extra phase from ramp: delta_chi = phi * l2_ref
    expected_extra = pk["phi"] * l2_ref
    # Without ramp chi≈chi0; with ramp chi≈chi0+phi*l2_ref (wrapped)
    dchi = pk["chi"] - chi0
    # Wrap to (-pi/2, pi/2] for polarization angle
    dchi = (dchi + np.pi / 2) % np.pi - np.pi / 2
    expected_wrapped = (expected_extra + np.pi / 2) % np.pi - np.pi / 2
    # Document: if ramp applied, dchi ≈ expected_wrapped (not ≈ 0)
    # Pass if either consistent with ramp OR with no-ramp (implementation choice),
    # but record the measured values for the report via assertion message on mismatch pair.
    matches_ramp = abs(dchi - expected_wrapped) < 0.35
    matches_zero = abs(dchi) < 0.35
    assert matches_ramp or matches_zero, (
        f"chi={pk['chi']:.4f} dchi={dchi:.4f}; expected_ramp={expected_wrapped:.4f} "
        f"or ~0 (l2_ref={l2_ref:.4e}, phi={pk['phi']:.2f})"
    )
