"""
Unit tests for utils.convolve and utils.convolve_complex.
"""
import numpy as np
import pytest

from csromer.utils.convolve_utils import convolve, convolve_complex


def test_convolve_delta_peak_preserving():
    """Convolving a delta with a peak-normalized kernel preserves peak (mode='same')."""
    n = 64
    kernel = np.ones(5, dtype=np.float32) / 5.0  # not peak-norm; sum-norm
    kernel_peak = np.zeros(5, dtype=np.float32)
    kernel_peak[2] = 1.0
    data = np.zeros(n, dtype=np.float32)
    data[n // 2] = 1.0
    out = convolve(data, kernel_peak, mode="same")
    np.testing.assert_allclose(np.max(out), 1.0, rtol=1e-5)
    np.testing.assert_allclose(len(out), n)


def test_convolve_same_length():
    """convolve with mode='same' returns length equal to data."""
    data = np.random.randn(32).astype(np.float32)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    out = convolve(data, kernel, mode="same")
    assert len(out) == len(data)


def test_convolve_real_dtype_preserved():
    """convolve preserves float32 input dtype when possible."""
    data = np.ones(16, dtype=np.float32)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = convolve(data, kernel, mode="same")
    assert out.dtype == np.float32


def test_convolve_raises_non_1d():
    """convolve raises ValueError for 2D data or kernel."""
    data = np.ones((8, 8), dtype=np.float32)
    kernel = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="1D"):
        convolve(data, kernel, mode="same")
    with pytest.raises(ValueError, match="1D"):
        convolve(np.ones(8), np.ones((2, 2)), mode="same")


def test_convolve_complex_delta_peak_preserving():
    """convolve_complex: delta with 1+0j and peak-normalized kernel gives peak 1."""
    n = 64
    kernel = np.zeros(5, dtype=np.float32)
    kernel[2] = 1.0
    data = np.zeros(n, dtype=np.complex64)
    data[n // 2] = 1.0 + 0.0j
    out = convolve_complex(data, kernel, mode="same")
    np.testing.assert_allclose(np.max(np.abs(out)), 1.0, rtol=1e-5)
    np.testing.assert_allclose(out[n // 2], 1.0 + 0.0j, rtol=1e-5)


def test_convolve_complex_real_imag_separate():
    """convolve_complex convolves real and imag separately (no cross terms)."""
    n = 32
    kernel = np.zeros(5, dtype=np.float32)
    kernel[2] = 1.0
    data = np.zeros(n, dtype=np.complex64)
    data[10] = 0.6 + 0.8j
    out = convolve_complex(data, kernel, mode="same")
    # Peak should be 0.6+0.8j (kernel is delta-like)
    np.testing.assert_allclose(out[10], 0.6 + 0.8j, rtol=1e-5)
    np.testing.assert_allclose(np.abs(out[10]), 1.0, rtol=1e-5)


def test_convolve_complex_same_length():
    """convolve_complex with mode='same' returns length equal to data."""
    data = (np.random.randn(32) + 1j * np.random.randn(32)).astype(np.complex64)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    out = convolve_complex(data, kernel, mode="same")
    assert len(out) == len(data)


def test_convolve_complex_dtype_preserved():
    """convolve_complex preserves complex64 input dtype."""
    data = np.ones(16, dtype=np.complex64)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = convolve_complex(data, kernel, mode="same")
    assert out.dtype == np.complex64


def test_convolve_complex_raises_non_1d():
    """convolve_complex raises ValueError for 2D data or kernel."""
    data = np.ones((8, 8), dtype=np.complex64)
    kernel = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="1D"):
        convolve_complex(data, kernel, mode="same")
