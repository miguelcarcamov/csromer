"""
Integration tests: run full reconstruction pipeline (simulate -> reconstruct -> check outputs).
Uses small data and few iterations so tests stay fast.
Requires optional dependency: pywt (PyWavelets).
"""
import numpy as np
import pytest

pytest.importorskip("pywt", reason="integration tests require PyWavelets")

from csromer.simulation import FaradayThinSource
from csromer.wrappers.reconstructors import CGReconstructorWrapper, CSROMERReconstructorWrapper


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
    """CG reconstructor: simulate -> reconstruct -> check shapes and finite outputs."""
    recon = CGReconstructorWrapper(
        dataset=small_thin_source,
        oversampling=4.0,
        cg_maxiter=5,
        cg_verbose=False,
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
