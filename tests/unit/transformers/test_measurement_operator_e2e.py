"""
End-to-end tests for measurement operators: conservation of energy (Parseval),
point source flux (1 Jy -> dirty value in Jy/RMTF), and RMTF normalization.
"""
import numpy as np
import pytest

from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.transformers.measurement_operator import DirectFourier1D, GriddedFFT1D
from csromer.utils.array_utils import asnumpy


def _norm2(arr):
    """Squared L2 norm (sum of |z|^2), works with numpy/dask."""
    s = np.sum(np.abs(arr)**2)
    return float(s.compute()) if hasattr(s,
                                         "compute") else float(np.sum(np.abs(np.asarray(arr))**2))


# ---- DirectFourier1D fixtures and tests ----


@pytest.fixture
def dataset_direct_fourier():
    """Dataset with uniform lambda² coverage and unit weights so K = n_ch."""
    n = 64
    lam2 = np.linspace(0.02, 0.1, n)**2
    d = Dataset(lambda2=lam2)
    d.w = np.ones(n, dtype=np.float64)
    d.data = np.ones(n, dtype=np.complex64)
    return d


@pytest.fixture
def parameter_direct_fourier(dataset_direct_fourier):
    """Parameter for Faraday depth grid (centered phi)."""
    p = Parameter()
    p.calculate_cellsize(dataset=dataset_direct_fourier, oversampling=4, verbose=False)
    return p


class TestDirectFourier1DE2E:
    """DirectFourier1D: energy conservation, point source flux, RMTF."""

    def test_energy_conservation_point_source(
        self, dataset_direct_fourier, parameter_direct_fourier
    ):
        """For a 1 Jy point source (delta), ||A x||^2 = n_ch (diagonal of A^H A)."""
        op = DirectFourier1D(dataset=dataset_direct_fourier, parameter=parameter_direct_fourier)
        n_phi = parameter_direct_fourier.n
        n_ch = dataset_direct_fourier.m
        center = n_phi // 2
        x = np.zeros(n_phi, dtype=np.complex64)
        x[center] = 1.0
        b = op.forward(x)
        energy_data = _norm2(b)
        np.testing.assert_allclose(energy_data, n_ch, rtol=1e-4)

    def test_energy_conservation_adjoint_forward(
        self, dataset_direct_fourier, parameter_direct_fourier
    ):
        """<x, A^H A x> = ||Ax||^2: adjoint(forward(x)) matches energy in data."""
        op = DirectFourier1D(dataset=dataset_direct_fourier, parameter=parameter_direct_fourier)
        n_phi = parameter_direct_fourier.n
        x = np.random.randn(n_phi).astype(np.complex64
                                          ) + 1j * np.random.randn(n_phi).astype(np.complex64)
        b = op.forward(x)
        a = op.adjoint(b)
        # <a, x> = <A^H A x, x> = ||Ax||^2 (real for this kernel)
        inner_val = np.real(np.vdot(a.ravel(), x.ravel()))
        inner = float(inner_val.compute()) if hasattr(inner_val,
                                                      "compute") else float(np.real(inner_val))
        energy_data = _norm2(b)
        np.testing.assert_allclose(inner, energy_data, rtol=1e-3)

    def test_point_source_1jy_dirty_value(self, dataset_direct_fourier, parameter_direct_fourier):
        """1 Jy point source at center -> dirty map peak = 1 (Jy per unit RMTF)."""
        op = DirectFourier1D(dataset=dataset_direct_fourier, parameter=parameter_direct_fourier)
        n_phi = parameter_direct_fourier.n
        center = n_phi // 2
        # 1 Jy point source at center
        x = np.zeros(n_phi, dtype=np.complex64)
        x[center] = 1.0
        data = op.forward(x)
        dataset_direct_fourier.data = data
        dirty = op.dirty_spectrum(data)
        dirty_np = np.asarray(asnumpy(dirty))
        peak = np.abs(dirty_np[center])
        # With K = n_ch, (A^H A delta)_center = n_ch, so dirty_peak = n_ch/K = 1
        np.testing.assert_allclose(peak, 1.0, rtol=1e-4)

    def test_rmtf_peak_normalization(self, dataset_direct_fourier, parameter_direct_fourier):
        """RMTF at center should be 1 when weights are uniform and K = n_ch."""
        op = DirectFourier1D(dataset=dataset_direct_fourier, parameter=parameter_direct_fourier)
        rmtf = op.RMTF()
        rmtf_np = np.asarray(asnumpy(rmtf))
        n_phi = parameter_direct_fourier.n
        center = n_phi // 2
        rmtf_peak = np.abs(rmtf_np[center])
        np.testing.assert_allclose(rmtf_peak, 1.0, rtol=1e-4)

    def test_point_source_dirty_equals_1_over_rmtf_peak(
        self, dataset_direct_fourier, parameter_direct_fourier
    ):
        """Dirty value at source = 1 Jy / RMTF_peak (same pixel); here both 1."""
        op = DirectFourier1D(dataset=dataset_direct_fourier, parameter=parameter_direct_fourier)
        n_phi = parameter_direct_fourier.n
        center = n_phi // 2
        x = np.zeros(n_phi, dtype=np.complex64)
        x[center] = 1.0
        data = op.forward(x)
        dataset_direct_fourier.data = data
        dirty = np.asarray(asnumpy(op.dirty_spectrum(data)))
        rmtf = np.asarray(asnumpy(op.RMTF()))
        dirty_peak = np.abs(dirty[center])
        rmtf_peak = np.abs(rmtf[center])
        # 1 Jy -> dirty_peak should be 1/RMTF_peak (so 1 when RMTF_peak=1)
        np.testing.assert_allclose(dirty_peak * rmtf_peak, 1.0, rtol=1e-4)


# ---- GriddedFFT1D fixtures and tests ----


@pytest.fixture
def dataset_gridded_fft():
    """Uniform lambda² and unit weights for GriddedFFT (K = n)."""
    n = 64
    lam2 = np.linspace(0.01, 0.1, n)**2
    d = Dataset(lambda2=lam2)
    d.w = np.ones(n, dtype=np.float64)
    d.data = np.ones(n, dtype=np.complex64)
    return d


@pytest.fixture
def parameter_gridded_fft(dataset_gridded_fft):
    """Parameter with phi matching FFT grid (e.g. linspace)."""
    n = 64
    p = Parameter(phi=np.linspace(-1, 1, n), data=np.zeros(n, dtype=np.complex64))
    return p


class TestGriddedFFT1DE2E:
    """GriddedFFT1D: Parseval, point source flux, RMTF."""

    def test_energy_conservation_parseval(self, dataset_gridded_fft, parameter_gridded_fft):
        """Parseval for FFT: sum |fft(x)|^2 = n * sum |x|^2."""
        op = GriddedFFT1D(dataset=dataset_gridded_fft, parameter=parameter_gridded_fft)
        n = parameter_gridded_fft.n
        x = np.random.randn(n).astype(np.complex64) + 1j * np.random.randn(n).astype(np.complex64)
        b = op.forward(x)
        energy_model = _norm2(x)
        energy_data = _norm2(b)
        np.testing.assert_allclose(energy_data, n * energy_model, rtol=1e-5)

    def test_point_source_1jy_dirty_value(self, dataset_gridded_fft, parameter_gridded_fft):
        """1 Jy point source (delta at center) -> dirty peak = 1 (Jy per unit RMTF), same as DirectFourier1D."""
        op = GriddedFFT1D(dataset=dataset_gridded_fft, parameter=parameter_gridded_fft)
        n = parameter_gridded_fft.n
        center = n // 2
        x = np.zeros(n, dtype=np.complex64)
        x[center] = 1.0
        data = op.forward(x)
        dataset_gridded_fft.data = data
        dirty = op.dirty_spectrum(data)
        dirty_np = np.asarray(asnumpy(dirty))
        peak = np.abs(dirty_np).max()
        # dirty_spectrum passes (w*p)/sum(w); adjoint then scaled by N to match direct FT. Peak = 1.
        expected_peak = 1.0
        np.testing.assert_allclose(peak, expected_peak, rtol=1e-5)

    def test_rmtf_peak(self, dataset_gridded_fft, parameter_gridded_fft):
        """GriddedFFT RMTF peak amplitude is 1."""
        op = GriddedFFT1D(dataset=dataset_gridded_fft, parameter=parameter_gridded_fft)
        rmtf = np.asarray(asnumpy(op.RMTF()))
        peak = np.abs(rmtf).max()
        np.testing.assert_allclose(peak, 1.0, rtol=1e-5)
