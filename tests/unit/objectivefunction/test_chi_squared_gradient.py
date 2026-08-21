"""
Unit tests for ChiSquared gradient with DFT (same operator as CG/dirty map).

Validates that the gradient matches a finite-difference approximation so FISTA
and CG use the same gradient when lambda=0.
"""
import numpy as np
import pytest

from csromer.objectivefunction import ChiSquared, OFunction
from csromer.reconstruction import Parameter
from csromer.simulation import FaradayThinSource
from csromer.transformers.dfts import NDFT1D


@pytest.fixture
def small_source_and_dft():
    """Minimal source, parameter, and DFT for ChiSquared gradient test."""
    nu = np.linspace(1.0e9, 1.5e9, 32)
    source = FaradayThinSource(nu=nu, s_nu=0.1, phi_gal=20.0, spectral_idx=0.0)
    source.simulate()
    param = Parameter()
    param.calculate_cellsize(dataset=source, oversampling=4.0, verbose=False)
    param.data = np.zeros(param.n, dtype=np.complex64)
    dft = NDFT1D(dataset=source, parameter=param)
    return source, param, dft


def test_chi_squared_gradient_vs_finite_difference(small_source_and_dft):
    """
    ChiSquared gradient with DFT should match finite-difference along any direction.

    With lambda=0, FISTA uses the same gradient as CG; this test ensures the
    gradient is correct.
    """
    source, param, dft = small_source_and_dft
    chi_squared = ChiSquared(measurement_operator=dft)
    F_obj = OFunction([chi_squared], persist_gradient=True)

    np.random.seed(42)
    x = np.zeros(param.n, dtype=np.complex128)
    # Small random point so we're in a smooth region
    x = (np.random.randn(param.n) + 1j * np.random.randn(param.n)).astype(np.complex128) * 0.01
    d = (np.random.randn(param.n) + 1j * np.random.randn(param.n)).astype(np.complex128)
    d = d / (np.sqrt(np.real(np.vdot(d.ravel(), d.ravel()))) + 1e-12)

    # Central difference: f'(0) ≈ (f(eps)-f(-eps))/(2*eps). Use eps that balances truncation and roundoff.
    eps = 1e-6
    f_plus = F_obj.evaluate(x + eps * d)
    f_minus = F_obj.evaluate(x - eps * d)
    fd_directional = (float(np.real(f_plus)) - float(np.real(f_minus))) / (2.0 * eps)

    grad = F_obj.calculate_gradient(x, differentiable_only=True)
    grad = np.asarray(grad)
    # Directional derivative Re(<grad, d>); numpy vdot(a,b) = sum conj(a)*b
    directional_derivative = np.real(np.vdot(grad.ravel(), d.ravel()))

    rel_err = abs(directional_derivative - fd_directional) / (abs(fd_directional) + 1e-14)
    assert rel_err < 3e-2, (
        f"ChiSquared gradient vs finite-diff: directional derivative {directional_derivative:.6e}, "
        f"fd {fd_directional:.6e}, rel err {rel_err:.6e}"
    )


def test_chi_squared_gradient_at_zero_proportional_to_dirty(small_source_and_dft):
    """
    At x=0, gradient of ChiSquared is -A^H W (Ax - b) = A^H W b (since Ax=0).
    Dirty spectrum is A^H(weighted data)/K, so gradient and dirty should be aligned.
    """
    source, param, dft = small_source_and_dft
    chi_squared = ChiSquared(measurement_operator=dft)
    x = np.zeros(param.n, dtype=np.complex64)
    grad = chi_squared.calculate_gradient(x)
    grad = np.asarray(grad)
    dirty = dft.dirty_spectrum(source.data)
    dirty = np.asarray(dirty)
    # Gradient at 0 = A^H W (0 - b) = -A^H W b; dirty = A^H(w*b/sum(w)) so same direction
    grad_norm = np.sqrt(np.real(np.vdot(grad.ravel(), grad.ravel()))) + 1e-14
    dirty_norm = np.sqrt(np.real(np.vdot(dirty.ravel(), dirty.ravel()))) + 1e-14
    cos_sim = np.real(np.vdot(grad.ravel(), dirty.ravel())) / (grad_norm * dirty_norm)
    # At x=0, grad = -A^H W b; dirty ∝ A^H W b, so grad and dirty are opposite (cos_sim ≈ -1)
    assert cos_sim < -0.99, (f"Gradient at 0 should be opposite to dirty (cos_sim={cos_sim:.4f})")
