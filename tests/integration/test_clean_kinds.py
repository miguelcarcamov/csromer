"""
Integration: φ-space vs major-cycle CLEAN recover point-source peak flux.
"""
import numpy as np
import pytest

pytest.importorskip("pywt", reason="CLEAN wrapper imports wavelet stack")

from csromer.pipelines.reconstruction import CLEANReconstructorWrapper
from csromer.simulation import FaradayThinSource
from csromer.utils.array_utils import asnumpy

pytestmark = pytest.mark.integration


def _window_peak(fd, phi, phi0, half_width):
    fd = np.asarray(asnumpy(fd))
    phi = np.asarray(asnumpy(phi))
    mask = np.abs(phi - phi0) < half_width
    idx = np.where(mask)[0][int(np.argmax(np.abs(fd[mask])))]
    return float(np.abs(fd[idx])), float(phi[idx])


def _thin_source(P0=1.0, phi0=50.0):
    nu = np.linspace(1.0e9, 1.5e9, 64)
    src = FaradayThinSource(nu=nu, s_nu=P0, phi_gal=phi0, spectral_idx=0.0)
    src.l2_ref = 0.0
    src.simulate()
    src.data = (P0 * np.exp(2.0j * (phi0 * np.asarray(src.lambda2)))).astype(
        np.complex64
    )
    return src


@pytest.mark.parametrize("clean_kind", ["phi", "major_cycle"])
def test_clean_kind_recovers_point_source_peak(clean_kind):
    P0, phi0 = 1.0, 50.0
    src = _thin_source(P0=P0, phi0=phi0)
    recon = CLEANReconstructorWrapper(
        dataset=src,
        oversampling=6.0,
        clean_kind=clean_kind,
        clean_maxiter=400,
        clean_gain=0.1,
        clean_threshold=1e-3,
    )
    recon.reconstruct()
    half = 3.0 * recon.parameter.rmtf_fwhm
    P, phi = _window_peak(recon.fd_restored, recon.parameter.phi, phi0, half)
    assert abs(phi - phi0) < recon.parameter.rmtf_fwhm
    assert abs(P - P0) / P0 < 0.12, (
        f"clean_kind={clean_kind}: restored peak {P:.4f} vs P0={P0}"
    )


def test_major_cycle_model_sparser_than_phi_on_oversampled_grid():
    """Major-cycle often concentrates flux into fewer δ pixels than RMTF Högbom."""
    P0, phi0 = 1.0, 50.0
    src_phi = _thin_source(P0=P0, phi0=phi0)
    src_maj = _thin_source(P0=P0, phi0=phi0)
    kwargs = dict(
        oversampling=8.0,
        clean_maxiter=300,
        clean_gain=0.1,
        clean_threshold=1e-3,
    )
    r_phi = CLEANReconstructorWrapper(dataset=src_phi, clean_kind="phi", **kwargs)
    r_phi.reconstruct()
    r_maj = CLEANReconstructorWrapper(dataset=src_maj, clean_kind="major_cycle", **kwargs)
    r_maj.reconstruct()

    nnz_phi = int(np.sum(np.abs(r_phi.fd_model) > 1e-4))
    nnz_maj = int(np.sum(np.abs(r_maj.fd_model) > 1e-4))
    P_phi, _ = _window_peak(
        r_phi.fd_restored, r_phi.parameter.phi, phi0, 3 * r_phi.parameter.rmtf_fwhm
    )
    P_maj, _ = _window_peak(
        r_maj.fd_restored, r_maj.parameter.phi, phi0, 3 * r_maj.parameter.rmtf_fwhm
    )
    assert abs(P_phi - P0) / P0 < 0.15
    assert abs(P_maj - P0) / P0 < 0.15
    # Soft check: major-cycle should not need more components than φ-RMTF
    assert nnz_maj <= nnz_phi + 5


def test_phi_and_major_cycle_results_similar():
    """
    Same noiseless thin source: both CLEAN kinds should agree on restored peak,
    φ peak location, and RM within tolerances (algorithms differ, science products match).
    """
    P0, phi0 = 1.0, 50.0
    kwargs = dict(
        oversampling=6.0,
        clean_maxiter=400,
        clean_gain=0.1,
        clean_threshold=1e-3,
    )
    r_phi = CLEANReconstructorWrapper(
        dataset=_thin_source(P0=P0, phi0=phi0), clean_kind="phi", **kwargs
    )
    r_phi.reconstruct()
    r_maj = CLEANReconstructorWrapper(
        dataset=_thin_source(P0=P0, phi0=phi0), clean_kind="major_cycle", **kwargs
    )
    r_maj.reconstruct()

    half = 3.0 * r_phi.parameter.rmtf_fwhm
    P_phi, phi_phi = _window_peak(r_phi.fd_restored, r_phi.parameter.phi, phi0, half)
    P_maj, phi_maj = _window_peak(r_maj.fd_restored, r_maj.parameter.phi, phi0, half)

    # Both recover injected flux and position
    assert abs(P_phi - P0) / P0 < 0.12
    assert abs(P_maj - P0) / P0 < 0.12
    assert abs(phi_phi - phi0) < r_phi.parameter.rmtf_fwhm
    assert abs(phi_maj - phi0) < r_maj.parameter.rmtf_fwhm

    # Cross-agreement: restored peaks within 10% of each other
    assert abs(P_phi - P_maj) / max(P_phi, P_maj) < 0.10, (
        f"restored peaks differ: phi={P_phi:.4f} major={P_maj:.4f}"
    )
    # Peak φ within one cell of each other (same grid setup)
    assert abs(phi_phi - phi_maj) <= max(r_phi.parameter.cellsize, r_maj.parameter.cellsize) * 1.5

    # Dirty maps should be identical (same data / operator path before CLEAN)
    np.testing.assert_allclose(
        np.asarray(asnumpy(r_phi.fd_dirty)),
        np.asarray(asnumpy(r_maj.fd_dirty)),
        rtol=1e-5,
        atol=1e-6,
    )

    # Restored spectra near the source agree in L2 (windowed)
    phi_grid = np.asarray(asnumpy(r_phi.parameter.phi))
    win = np.abs(phi_grid - phi0) < half
    rest_phi = np.asarray(asnumpy(r_phi.fd_restored))[win]
    rest_maj = np.asarray(asnumpy(r_maj.fd_restored))[win]
    # Align lengths if grids differ slightly (should not with same oversampling)
    assert rest_phi.shape == rest_maj.shape
    rel_l2 = float(
        np.linalg.norm(rest_phi - rest_maj) / (np.linalg.norm(rest_phi) + 1e-30)
    )
    assert rel_l2 < 0.20, (
        f"windowed restored L2 relative difference {rel_l2:.3f} too large "
        f"(P_phi={P_phi:.4f}, P_maj={P_maj:.4f})"
    )
