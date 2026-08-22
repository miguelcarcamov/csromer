"""
Unit tests for the TV and TSV regularization terms.

These files sat at 0% coverage, which is why several defects accumulated in them
(see issue #28): a TSV gradient that used the TV sign-based subgradient formula, an
is_differentiable flag contradicting its own docstring, and a `nu` argument ignored
by both proximal operators.
"""
import numpy as np
import pytest

pytest.importorskip("prox_tv", reason="TV/TSV proximal operators require prox_tv")

from csromer.objectivefunction.priors.tsv import TSV
from csromer.objectivefunction.priors.tv import TV


def _finite_difference_gradient(term, x, h=1e-6):
    """Central-difference gradient of term.evaluate, for real input."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp_, xm_ = x.copy(), x.copy()
        xp_[i] += h
        xm_[i] -= h
        grad[i] = (float(term.evaluate(xp_)) - float(term.evaluate(xm_))) / (2 * h)
    return grad


class TestEvaluate:

    def test_tsv_matches_explicit_sum(self):
        x = np.array([1.0, 3.0, 2.0, 6.0])
        # differences 2, -1, 4 -> 4 + 1 + 16
        assert float(TSV(reg=1.0).evaluate(x)) == pytest.approx(21.0)

    def test_tv_matches_explicit_sum(self):
        x = np.array([1.0, 3.0, 2.0, 6.0])
        # |2| + |-1| + |4|
        assert float(TV(reg=1.0).evaluate(x)) == pytest.approx(7.0)

    def test_constant_signal_has_zero_variation(self):
        x = np.full(10, 2.5)
        assert float(TSV(reg=1.0).evaluate(x)) == pytest.approx(0.0)
        assert float(TV(reg=1.0).evaluate(x)) == pytest.approx(0.0)

    def test_complex_uses_modulus(self):
        x = np.array([0.0, 3.0 + 4.0j])  # single difference of modulus 5
        assert float(TV(reg=1.0).evaluate(x)) == pytest.approx(5.0)
        assert float(TSV(reg=1.0).evaluate(x)) == pytest.approx(25.0)


class TestTSVGradient:
    """TSV is smooth, so its gradient must match finite differences exactly."""

    def test_gradient_matches_finite_differences(self):
        rng = np.random.RandomState(0)
        x = rng.randn(12)
        term = TSV(reg=1.0)
        analytic = np.asarray(term.calculate_gradient(x), dtype=float)
        numeric = _finite_difference_gradient(term, x)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    def test_gradient_includes_endpoints(self):
        """The previous implementation left both endpoints at zero."""
        x = np.array([0.0, 1.0, 2.0, 5.0])
        grad = np.asarray(TSV(reg=1.0).calculate_gradient(x), dtype=float)
        assert grad[0] != 0.0
        assert grad[-1] != 0.0

    def test_gradient_is_not_sign_quantized(self):
        """
        Regression guard for #28: the old implementation used np.sign, so its output
        was quantized to multiples of 2 regardless of how large the differences were.
        """
        x = np.array([0.0, 10.0, 20.0])
        grad = np.asarray(TSV(reg=1.0).calculate_gradient(x), dtype=float)
        assert np.max(np.abs(grad)) > 4.0

    def test_constant_signal_has_zero_gradient(self):
        x = np.full(8, 3.0)
        grad = np.asarray(TSV(reg=1.0).calculate_gradient(x), dtype=float)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)


class TestTVGradient:

    def test_subgradient_reduces_to_sign_for_real_input(self):
        x = np.array([0.0, 1.0, 0.0])
        grad = np.asarray(TV(reg=1.0).calculate_gradient(x), dtype=float)
        # interior point sits at a peak: +1 from the left difference, +1 from the right
        assert grad[1] == pytest.approx(2.0, abs=1e-3)

    def test_constant_signal_has_zero_subgradient(self):
        x = np.full(8, 3.0)
        grad = np.asarray(TV(reg=1.0).calculate_gradient(x), dtype=float)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_complex_input_returns_complex_subgradient(self):
        x = np.array([0.0, 1.0 + 1.0j, 0.0], dtype=np.complex128)
        grad = np.asarray(TV(reg=1.0).calculate_gradient(x))
        assert np.iscomplexobj(grad)
        assert grad.shape == x.shape


class TestDifferentiabilityFlags:

    def test_tsv_is_differentiable(self):
        """TSV is a sum of squared differences, hence smooth. Regression for #28."""
        assert TSV(reg=1.0).is_differentiable is True

    def test_tv_is_not_differentiable(self):
        assert TV(reg=1.0).is_differentiable is False


class TestProxHonoursNu:
    """
    Regression for #28: both proxes ignored `nu` and always thresholded on `reg`.
    FISTA varies its step size, so a prox that ignores nu solves the wrong problem.
    """

    @pytest.mark.parametrize("term_cls", [TV, TSV])
    def test_nu_scales_the_threshold(self, term_cls):
        rng = np.random.RandomState(2)
        x = rng.randn(64)
        term = term_cls(reg=0.1)
        weak = np.asarray(term.calculate_prox(x, nu=1.0))
        strong = np.asarray(term.calculate_prox(x, nu=10.0))
        # A larger nu means heavier regularization, hence less remaining variation.
        assert float(term.evaluate(strong)) < float(term.evaluate(weak))

    @pytest.mark.parametrize("term_cls", [TV, TSV])
    def test_nu_zero_falls_back_to_reg(self, term_cls):
        rng = np.random.RandomState(3)
        x = rng.randn(32)
        term = term_cls(reg=0.5)
        np.testing.assert_allclose(
            np.asarray(term.calculate_prox(x, nu=0.0)),
            np.asarray(term.calculate_prox(x, nu=1.0)),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("term_cls", [TV, TSV])
    def test_prox_accepts_complex_input(self, term_cls):
        rng = np.random.RandomState(4)
        x = (rng.randn(32) + 1j * rng.randn(32)).astype(np.complex64)
        out = np.asarray(term_cls(reg=0.1).calculate_prox(x, nu=1.0))
        assert np.iscomplexobj(out)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))


class TestTSVHasNoIsotropyDistinction:
    """
    |d|^2 == Re(d)^2 + Im(d)^2 exactly, so isotropic and anisotropic TSV coincide and
    TSV is phase-rotation invariant for free. TV has no such property.
    """

    def test_tsv_is_invariant_under_global_phase_rotation(self):
        rng = np.random.RandomState(5)
        x = rng.randn(64) + 1j * rng.randn(64)
        term = TSV(reg=1.0)
        base = float(term.evaluate(x))
        for deg in (15, 45, 90, 180):
            rotated = x * np.exp(1j * np.deg2rad(deg))
            assert float(term.evaluate(rotated)) == pytest.approx(base, rel=1e-10)

    def test_isotropic_tv_is_invariant_under_global_phase_rotation(self):
        rng = np.random.RandomState(6)
        x = rng.randn(64) + 1j * rng.randn(64)
        term = TV(reg=1.0)
        base = float(term.evaluate(x))
        for deg in (15, 45, 90):
            rotated = x * np.exp(1j * np.deg2rad(deg))
            assert float(term.evaluate(rotated)) == pytest.approx(base, rel=1e-10)
