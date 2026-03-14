"""
Integration tests: run full reconstruction pipeline (simulate -> reconstruct -> check outputs).
Uses small data and few iterations so tests stay fast.
Requires PyWavelets (pywt). Install with: pip install PyWavelets
"""
import numpy as np
import pytest

pytest.importorskip("pywt", reason="integration tests require PyWavelets (pip install PyWavelets)")


pytestmark = pytest.mark.integration

from csromer.simulation import FaradayThinSource
from csromer.pipelines.reconstruction import (
    CLEANReconstructorWrapper,
    CSROMERReconstructorWrapper,
    make_cg_optimizer,
    make_fista_optimizer,
)


@pytest.fixture
def small_thin_source():
    """Minimal simulated thin source for fast integration test."""
    nu = np.linspace(1.0e9, 1.5e9, 64)
    source = FaradayThinSource(
        nu=nu,
        s_nu=0.01,
        phi_gal=10.0,
        spectral_idx=0.0,
    )
    source.simulate()
    return source


def test_cg_reconstruction_end_to_end(small_thin_source):
    """CG path via CSROMER: simulate -> reconstruct -> check shapes and finite outputs."""
    recon = CSROMERReconstructorWrapper(
        dataset=small_thin_source,
        oversampling=4.0,
        optimizer_factory=make_cg_optimizer(maxiter=5, verbose=False),
    )
    recon.reconstruct()

    n_phi = recon.parameter.phi.shape[0]
    assert recon.fd_dirty.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_dirty))
    assert recon.fd_model.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_model))
    assert recon.fd_restored.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_restored))
    assert recon.fd_residual.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_residual))

    assert np.isfinite(recon.rm_dirty)
    assert np.isfinite(recon.rm_model)
    assert np.isfinite(recon.rm_restored)
    assert np.isfinite(recon.second_moment)


def test_clean_reconstruction_end_to_end(small_thin_source):
    """CLEAN reconstructor: simulate -> reconstruct -> check shapes and finite outputs."""
    recon = CLEANReconstructorWrapper(
        dataset=small_thin_source,
        oversampling=4.0,
        clean_maxiter=50,
        clean_gain=0.2,
    )
    recon.reconstruct()

    n_phi = recon.parameter.phi.shape[0]
    assert recon.fd_dirty.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_dirty))
    assert recon.fd_model.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_model))
    assert recon.fd_restored.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_restored))
    assert recon.fd_residual.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_residual))
    assert np.isfinite(recon.rm_dirty)
    assert np.isfinite(recon.rm_model)
    assert np.isfinite(recon.rm_restored)
    assert np.isfinite(recon.second_moment)


def test_fista_reconstruction_end_to_end(small_thin_source):
    """FISTA (CSROMER) reconstructor: simulate -> reconstruct -> check shapes and finite outputs."""
    recon = CSROMERReconstructorWrapper(
        dataset=small_thin_source,
        oversampling=4.0,
        wavelet=None,
    )
    recon.reconstruct()

    n_phi = recon.parameter.phi.shape[0]
    assert recon.fd_dirty.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_dirty))
    assert recon.fd_model.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_model))
    assert recon.fd_restored.shape == (n_phi,)
    assert np.all(np.isfinite(recon.fd_restored))
    assert np.isfinite(recon.rm_model)
    assert np.isfinite(recon.second_moment)


@pytest.mark.xfail(reason="FISTA currently stuck in monotone reject; model/restored near zero until fixed")
def test_fista_restored_amplitude_vs_dirty(small_thin_source):
    """
    FISTA with lambda_l_norm=0 (Chi-squared only, same as CG) should produce
    non-trivial model/restored: max|restored| should be a non-negligible
    fraction of max|dirty| (restored close to dirty in scale).
    Remove xfail when FISTA backtracking/step is fixed.
    """
    recon = CSROMERReconstructorWrapper(
        dataset=small_thin_source,
        oversampling=4.0,
        wavelet=None,
        lambda_l_norm=0.0,
        optimizer_factory=make_fista_optimizer(maxiter=100, verbose=False),
    )
    recon.reconstruct()

    max_dirty = np.max(np.abs(recon.fd_dirty))
    max_restored = np.max(np.abs(recon.fd_restored))
    max_model = np.max(np.abs(recon.fd_model))
    assert max_dirty > 0, "Dirty spectrum should have non-zero peak"
    # Restored = conv(model) * rmtf_fwhm + residual; should be on same scale as dirty
    assert max_restored >= 1e-6 * max_dirty, (
        f"FISTA restored should have non-negligible amplitude: "
        f"max|restored|={max_restored:.2e}, max|dirty|={max_dirty:.2e}"
    )
    # Model (Jy/phi_pixel) can be smaller than dirty (Jy/rmtf); check it's not identically zero
    assert max_model >= 1e-10, (
        f"FISTA model should be non-zero: max|model|={max_model:.2e}"
    )
