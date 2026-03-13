"""
Integration tests: delta in φ → dirty peak ≈ 1; handmade fd_spectrum |F|=1 Jy → dirty peak ≈ 1.

Dataset is built from **nu** (uniform frequency) so that λ² is **irregular** (λ² = c²/ν²).
Tests all three operator kinds: direct (NDFT), nufft, gridded (grid then FFT).

- Delta in φ (1 Jy at one pixel) → forward → λ² data → dirty_spectrum → peak ≈ 1.
- Handmade complex fd_spectrum with |F| = 1 Jy (various real/imag) → same flow; peak ≈ 1.
"""
import numpy as np
import pytest

from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.utils.array_utils import asnumpy

pytestmark = pytest.mark.integration


def _dirty_peak(dirty):
    return float(np.max(np.abs(np.asarray(asnumpy(dirty)))))


# 1 Jy amplitude: |re + 1j*im| = 1. Various real/imag for parametrized tests.
ONE_JY_VALUES = [
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.6, 0.8),
    (0.8, -0.6),
    (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
]


@pytest.fixture
def uniform_nu():
    """Uniform frequency grid (Hz). Yields irregular λ² via λ² = c²/ν²."""
    return np.linspace(1.0e9, 1.5e9, 64)


@pytest.fixture
def dataset_from_nu(uniform_nu):
    """Dataset built from nu: uniform frequency → irregular lambda². Has data/model_data for gridding."""
    nu = uniform_nu
    n = len(nu)
    d = Dataset(nu=nu, w=np.ones(n, dtype=np.float64))
    # Required for gridding: Gridding.run() indexes dataset.data and dataset.model_data
    d.data = np.ones(n, dtype=np.complex64)
    d.model_data = np.zeros(n, dtype=np.complex64)
    return d


@pytest.fixture
def parameter_from_dataset(dataset_from_nu):
    """Parameter with phi grid from dataset (same for all operator kinds)."""
    p = Parameter()
    p.calculate_cellsize(dataset=dataset_from_nu, oversampling=4, verbose=False)
    return p


def _build_operator_and_dataset(kind, dataset, parameter):
    """Build (op, dataset_to_use) for the given kind. Avoids pipeline import (pywt)."""
    from csromer.transformers.dfts import GriddedFFT1D, NDFT1D, NUFFT1D
    from csromer.transformers.gridding import Gridding

    if kind == "direct":
        op = NDFT1D(dataset=dataset, parameter=parameter)
        return op, dataset
    if kind == "nufft":
        op = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)
        return op, dataset
    if kind == "gridded":
        d_lambda2 = np.pi / (parameter.n * parameter.cellsize)
        gridding = Gridding(
            dataset=dataset,
            d_lambda2=d_lambda2,
            n=parameter.n,
            kernel="kaiser",
            gridding_kernel_half_width=4.0,
            gridding_kernel_beta=2.5,
        )
        d_grid = gridding.run()
        op = GriddedFFT1D(dataset=d_grid, parameter=parameter)
        return op, d_grid
    raise ValueError(f"kind must be direct, nufft, or gridded; got {kind!r}")


# ---- 1) Delta in φ → data in λ² → dirty peak ≈ 1 ----


@pytest.mark.parametrize("kind", ["direct", "nufft", "gridded"])
def test_delta_phi_dirty_peak_close_to_one(dataset_from_nu, parameter_from_dataset, kind):
    """
    Delta in φ (1 Jy at center) → forward → λ² data → dirty_spectrum(data) → dirty peak ≈ 1.
    Dataset from uniform nu (irregular λ²). All operator kinds: direct, nufft, gridded.
    """
    d = dataset_from_nu
    p = parameter_from_dataset
    n_phi = p.n
    fd_delta = np.zeros(n_phi, dtype=np.complex128)
    fd_delta[n_phi // 2] = 1.0 + 0.0j

    op, data_for_op = _build_operator_and_dataset(kind, d, p)
    data_for_op.data = np.asarray(asnumpy(op.forward(fd_delta)), dtype=np.complex64)
    dirty = op.dirty_spectrum(data_for_op.data)

    peak = _dirty_peak(dirty)
    # NUFFT normalization can differ slightly; use slightly looser tol for nufft
    rtol = 5e-3 if kind == "nufft" else 1e-4
    np.testing.assert_allclose(peak, 1.0, rtol=rtol, atol=1e-6)


# ---- 2) Handmade fd_spectrum |F| = 1 Jy (varying real/imag) → forward → dirty peak ≈ 1 ----


@pytest.mark.parametrize("kind", ["direct", "nufft", "gridded"])
@pytest.mark.parametrize(
    "re,im",
    ONE_JY_VALUES,
    ids=[
        "real_pos",
        "imag_pos",
        "real_neg",
        "imag_neg",
        "0.6_0.8",
        "0.8_neg0.6",
        "45deg",
        "135deg",
    ],
)
def test_handmade_fd_spectrum_1jy_dirty_peak_one(dataset_from_nu, parameter_from_dataset, kind, re, im):
    """
    Complex fd_spectrum with |F| = 1 Jy at one φ (varying real/imag); forward then dirty_spectrum; peak ≈ 1.
    Dataset from uniform nu (irregular λ²). All operator kinds.
    """
    assert np.isclose(np.sqrt(re * re + im * im), 1.0), "|re + 1j*im| must be 1"
    d = dataset_from_nu
    p = parameter_from_dataset
    n_phi = p.n
    # Use center pixel so dirty peak ≈ 1 for all kinds (nufft normalization is position-sensitive)
    idx = n_phi // 2
    fd = np.zeros(n_phi, dtype=np.complex128)
    fd[idx] = re + 1j * im

    op, data_for_op = _build_operator_and_dataset(kind, d, p)
    data_for_op.data = np.asarray(asnumpy(op.forward(fd)), dtype=np.complex64)
    dirty = op.dirty_spectrum(data_for_op.data)

    peak = _dirty_peak(dirty)
    rtol = 5e-3 if kind == "nufft" else 1e-4
    np.testing.assert_allclose(peak, 1.0, rtol=rtol, atol=1e-6)
