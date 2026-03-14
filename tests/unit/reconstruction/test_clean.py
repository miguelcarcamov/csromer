"""
Unit tests for 1D CLEAN algorithm (clean_1d and _shift_rmtf_to_peak).

We import via the public pipelines API (csromer.pipelines.reconstruction.clean),
even though this requires PyWavelets (pywt) to be installed in the environment.
"""
import numpy as np
import pytest

from csromer.pipelines.reconstruction.clean import clean_1d, _shift_rmtf_to_peak


@pytest.fixture
def n_phi():
    return 64


@pytest.fixture
def rmtf_gaussian_like(n_phi):
    """RMTF-like kernel with peak at center (Gaussian shape)."""
    x = np.arange(n_phi, dtype=np.float64) - n_phi // 2
    sigma = 3.0
    r = np.exp(-0.5 * (x / sigma) ** 2)
    return (r + 0.01 * np.random.RandomState(42).randn(n_phi)).astype(np.complex128)


def test_shift_rmtf_to_peak(n_phi, rmtf_gaussian_like):
    """Shifted RMTF has its maximum at the requested peak index."""
    rmtf0 = rmtf_gaussian_like
    center = n_phi // 2
    peak_idx = 10
    shifted = _shift_rmtf_to_peak(rmtf0, peak_idx, n_phi)
    assert shifted.shape == rmtf0.shape
    imax = int(np.argmax(np.abs(shifted)))
    assert imax == peak_idx


def test_shift_rmtf_to_peak_center_unchanged(n_phi, rmtf_gaussian_like):
    """Shift by center index leaves peak at center (roll 0)."""
    rmtf0 = rmtf_gaussian_like
    center = n_phi // 2
    shifted = _shift_rmtf_to_peak(rmtf0, center, n_phi)
    np.testing.assert_array_almost_equal(shifted, rmtf0)


def test_clean_1d_single_component(n_phi, rmtf_gaussian_like):
    """Dirty = single delta; CLEAN should put model component at that pixel."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    idx = n_phi // 2
    dirty[idx] = 1.0 + 0.0j
    gain = 0.2
    model, residual = clean_1d(dirty, rmtf_gaussian_like, gain=gain, maxiter=20)
    assert model.shape == dirty.shape
    assert residual.shape == dirty.shape
    assert np.argmax(np.abs(model)) == idx
    assert np.abs(model[idx]) > 0
    # Residual peak should be lower than initial
    assert np.max(np.abs(residual)) < np.max(np.abs(dirty)) + 1e-10


def test_clean_1d_residual_decreases(n_phi, rmtf_gaussian_like):
    """Over iterations residual peak should decrease (manual loop check)."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    dirty[n_phi // 2] = 1.0 + 0.0j
    model1, res1 = clean_1d(dirty, rmtf_gaussian_like, gain=0.2, maxiter=2)
    model2, res2 = clean_1d(dirty, rmtf_gaussian_like, gain=0.2, maxiter=5)
    assert np.max(np.abs(res2)) <= np.max(np.abs(res1)) + 1e-10


def test_clean_1d_maxiter_stop(n_phi, rmtf_gaussian_like):
    """Loop stops at maxiter; shapes unchanged."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    dirty[n_phi // 2] = 1.0
    model, residual = clean_1d(
        dirty, rmtf_gaussian_like, gain=0.2, maxiter=3, threshold=None
    )
    assert model.shape == (n_phi,)
    assert residual.shape == (n_phi,)
    # Should have done exactly 3 components (or fewer if residual went to zero)
    n_comp = np.sum(np.abs(model) > 1e-12)
    assert n_comp <= 3


def test_clean_1d_threshold_stop(n_phi, rmtf_gaussian_like):
    """When threshold is high, loop stops before maxiter."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    dirty[n_phi // 2] = 0.5 + 0.0j  # small peak
    threshold = 0.4  # above final residual after one iteration
    model, residual = clean_1d(
        dirty, rmtf_gaussian_like, gain=0.2, maxiter=100, threshold=threshold
    )
    assert np.max(np.abs(residual)) < threshold + 0.1  # stopped due to threshold


def test_clean_1d_complex(n_phi, rmtf_gaussian_like):
    """Complex dirty and RMTF; model and residual are complex and consistent."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    idx = n_phi // 2
    dirty[idx] = 0.6 + 0.8j  # 1 Jy amplitude
    rmtf_complex = rmtf_gaussian_like * (1.0 + 0.0j)
    model, residual = clean_1d(dirty, rmtf_complex, gain=0.2, maxiter=10)
    assert np.iscomplexobj(model)
    assert np.iscomplexobj(residual)
    assert model.dtype == np.complex128
    assert np.argmax(np.abs(model)) == idx
    # For one component: residual ≈ dirty - gain * peak * rmtf_shifted
    # So |residual| should be smaller than |dirty| after first iteration
    assert np.max(np.abs(residual)) < np.max(np.abs(dirty)) + 1e-10


def test_clean_1d_mismatch_length_raises(n_phi, rmtf_gaussian_like):
    """dirty and rmtf_at_zero must have same length."""
    dirty = np.zeros(n_phi + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="same length"):
        clean_1d(dirty, rmtf_gaussian_like, gain=0.2, maxiter=5)


def test_clean_1d_zero_rmtf_returns_zero_model(n_phi):
    """If RMTF is zero, return zero model and dirty as residual."""
    dirty = np.zeros(n_phi, dtype=np.complex128)
    dirty[n_phi // 2] = 1.0
    rmtf_zero = np.zeros(n_phi, dtype=np.complex128)
    model, residual = clean_1d(dirty, rmtf_zero, gain=0.2, maxiter=10)
    np.testing.assert_array_almost_equal(model, 0.0)
    np.testing.assert_array_almost_equal(residual, dirty)
