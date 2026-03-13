"""
Unit tests for Parameter.convolve: peak-conserving clean beam.

An fd_spectrum with peak |F| = 1 Jy at some φ, convolved with the clean RMTF Gaussian,
should have the same peak (peak-conserving restoration).
Tests several complex combinations (real/imag) so that amplitude is 1 Jy.
"""
import numpy as np
import pytest

from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.utils.array_utils import asnumpy


@pytest.fixture
def dataset_for_parameter():
    """Dataset from uniform frequency so λ² is irregular (realistic)."""
    nu = np.linspace(1.0e9, 1.5e9, 64)
    d = Dataset(nu=nu, w=np.ones(len(nu), dtype=np.float64))
    return d


@pytest.fixture
def parameter_with_grid(dataset_for_parameter):
    """Parameter with phi grid, cellsize and rmtf_fwhm from dataset (irregular λ²)."""
    p = Parameter()
    p.calculate_cellsize(dataset=dataset_for_parameter, oversampling=4, verbose=False)
    return p


# 1 Jy amplitude = |re + 1j*im| = 1. Various real/imag combinations.
ONE_JY_DELTA_VALUES = [
    pytest.param(1.0, 0.0, id="real_positive"),
    pytest.param(0.0, 1.0, id="imag_positive"),
    pytest.param(-1.0, 0.0, id="real_negative"),
    pytest.param(0.0, -1.0, id="imag_negative"),
    pytest.param(0.6, 0.8, id="mixed_0.6_0.8"),
    pytest.param(0.8, -0.6, id="mixed_0.8_neg0.6"),
    pytest.param(1.0 / np.sqrt(2), 1.0 / np.sqrt(2), id="45deg"),
    pytest.param(-1.0 / np.sqrt(2), 1.0 / np.sqrt(2), id="135deg"),
]


@pytest.mark.parametrize("re,im", ONE_JY_DELTA_VALUES)
def test_convolve_conserves_peak_for_delta_1jy(parameter_with_grid, re, im):
    """
    FD spectrum with peak |F| = 1 Jy at one pixel (varying real/imag),
    convolved with clean RMTF Gaussian, should have the same peak (peak-conserving).
    """
    p = parameter_with_grid
    n = p.n
    center = n // 2
    val = re + 1j * im
    assert np.isclose(np.abs(val), 1.0), "Test assumes |val| = 1"
    fd_delta = np.zeros(n, dtype=np.complex128)
    fd_delta[center] = val
    conv = p.convolve(x=fd_delta, rmtf_fwhm=p.rmtf_fwhm)
    conv_np = np.asarray(asnumpy(conv))
    peak_out = float(np.max(np.abs(conv_np)))
    np.testing.assert_allclose(peak_out, 1.0, rtol=1e-5, atol=1e-7)


def test_convolve_conserves_peak_for_delta_at_arbitrary_index(parameter_with_grid):
    """Peak conservation when delta is not at center."""
    p = parameter_with_grid
    n = p.n
    idx = n // 4
    fd_delta = np.zeros(n, dtype=np.complex128)
    fd_delta[idx] = 1.0 + 0.0j
    conv = p.convolve(x=fd_delta, rmtf_fwhm=p.rmtf_fwhm)
    conv_np = np.asarray(asnumpy(conv))
    peak_out = float(np.max(np.abs(conv_np)))
    np.testing.assert_allclose(peak_out, 1.0, rtol=1e-5, atol=1e-7)
