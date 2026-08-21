"""
Tests for measurement operators: forward/adjoint shapes and consistency,
and dask array support for DirectFourier1D.
"""
import numpy as np
import pytest

from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.transformers.measurement_operator import (
    NUFFT1D,
    DirectFourier1D,
    GriddedFFT1D,
    MeasurementOperator,
)


@pytest.fixture
def dataset_numpy():
    """Small dataset with numpy arrays."""
    lam2 = np.linspace(0.02, 0.1, 32)**2
    d = Dataset(lambda2=lam2)
    d.data = np.ones(32, dtype=np.complex64)
    return d


@pytest.fixture
def parameter_numpy(dataset_numpy):
    """Parameter for Faraday depth space."""
    p = Parameter()
    p.calculate_cellsize(dataset=dataset_numpy, oversampling=4, verbose=False)
    return p


class TestDirectFourier1D:
    """DirectFourier1D forward/adjoint and backward compatibility."""

    def test_forward_adjoint_shapes(self, dataset_numpy, parameter_numpy):
        op = DirectFourier1D(dataset=dataset_numpy, parameter=parameter_numpy)
        n_phi = parameter_numpy.n
        n_chan = dataset_numpy.m
        x = np.zeros(n_phi, dtype=np.complex64)
        x[n_phi // 2] = 1.0
        b = op.forward(x)
        assert b.shape == (n_chan, )
        a = op.adjoint(b)
        assert a.shape == (n_phi, )

    def test_backward_equals_adjoint(self, dataset_numpy, parameter_numpy):
        op = DirectFourier1D(dataset=dataset_numpy, parameter=parameter_numpy)
        n_chan = dataset_numpy.m
        b = np.ones(n_chan, dtype=np.complex64) / n_chan
        np.testing.assert_allclose(op.adjoint(b), op.backward(b))

    def test_adjoint_forward_composition(self, dataset_numpy, parameter_numpy):
        op = DirectFourier1D(dataset=dataset_numpy, parameter=parameter_numpy)
        n_phi = parameter_numpy.n
        x = np.random.randn(n_phi).astype(np.float32
                                          ) + 1j * np.random.randn(n_phi).astype(np.float32)
        x = x.astype(np.complex64)
        b = op.forward(x)
        a = op.adjoint(b)
        assert a.shape == x.shape
        assert np.isfinite(a).all()
        assert np.linalg.norm(a) > 0

    def test_rmtf_shape(self, dataset_numpy, parameter_numpy):
        op = DirectFourier1D(dataset=dataset_numpy, parameter=parameter_numpy)
        rmtf = op.RMTF()
        assert rmtf.shape == (parameter_numpy.n, )


class TestDirectFourier1DDask:
    """DirectFourier1D with dask array inputs (no unnecessary numpy conversion)."""

    @pytest.mark.skipif(
        __import__("sys").modules.get("dask") is None,
        reason="dask not installed",
    )
    def test_forward_adjoint_dask(self, dataset_numpy, parameter_numpy):
        import dask.array as da

        # Use dask for lambda2 so operator uses dask path
        lam2 = da.from_array(np.linspace(0.02, 0.1, 32)**2, chunks=16)
        d = Dataset(lambda2=lam2)
        d.data = da.ones(32, dtype=np.complex64, chunks=16)
        p = Parameter()
        p.calculate_cellsize(dataset=d, oversampling=4, verbose=False)
        op = DirectFourier1D(dataset=d, parameter=p)
        n_phi = p.n
        x_np = np.zeros(n_phi, dtype=np.complex64)
        x_np[n_phi // 2] = 1.0
        x = da.from_array(x_np, chunks=n_phi // 2)
        b = op.forward(x)
        assert hasattr(b, "compute")  # dask array
        assert b.shape == (32, )
        a = op.adjoint(b)
        assert hasattr(a, "compute")
        assert a.shape == (n_phi, )


class TestGriddedFFT1D:
    """GriddedFFT1D forward/adjoint using fft/ifft."""

    def test_forward_adjoint_roundtrip(self):
        n = 64
        lam2 = np.linspace(0.01, 0.1, n)**2
        d = Dataset(lambda2=lam2)
        p = Parameter(phi=np.linspace(-1, 1, n), data=np.zeros(n, dtype=np.complex64))
        op = GriddedFFT1D(dataset=d, parameter=p)
        x = np.random.randn(n).astype(np.complex64)
        b = op.forward(x)
        a = op.adjoint(b)
        np.testing.assert_allclose(a, x, rtol=1e-5, atol=1e-6)

    def test_backward_equals_adjoint(self):
        n = 32
        lam2 = np.linspace(0.01, 0.05, n)**2
        d = Dataset(lambda2=lam2)
        p = Parameter(phi=np.linspace(-1, 1, n), data=np.zeros(n, dtype=np.complex64))
        op = GriddedFFT1D(dataset=d, parameter=p)
        b = np.ones(n, dtype=np.complex64)
        np.testing.assert_allclose(op.adjoint(b), op.backward(b))


class TestNUFFT1D:
    """NUFFT1D: Hilbert adjoint matches forward (same check as for exact NDFT / gridded FFT)."""

    def test_hilbert_adjoint_identity(self, dataset_numpy, parameter_numpy):
        """⟨A x, y⟩ = ⟨x, Aᴴ y⟩ with numpy.vdot (standard complex Hilbert pairing)."""
        op = NUFFT1D(dataset=dataset_numpy, parameter=parameter_numpy)
        n_phi = parameter_numpy.n
        n_chan = dataset_numpy.m
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n_phi) + 1j * rng.standard_normal(n_phi)
        x = x.astype(np.complex64)
        y = rng.standard_normal(n_chan) + 1j * rng.standard_normal(n_chan)
        y = y.astype(np.complex64)
        lhs = np.vdot(op.forward(x), y)
        rhs = np.vdot(x, op.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)

    def test_backward_equals_adjoint(self, dataset_numpy, parameter_numpy):
        op = NUFFT1D(dataset=dataset_numpy, parameter=parameter_numpy)
        n_chan = dataset_numpy.m
        b = np.ones(n_chan, dtype=np.complex64) / n_chan
        np.testing.assert_allclose(op.adjoint(b), op.backward(b))
