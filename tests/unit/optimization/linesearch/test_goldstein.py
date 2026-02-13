"""Unit tests for Goldstein line search (lower and upper bound)."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import Goldstein


class TestGoldstein:
    """Test Goldstein: f(x)+c2*alpha*grad'd <= f(x+alpha*d) <= f(x)+c1*alpha*grad'd."""

    def test_goldstein_returns_valid_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Goldstein(step=1.0, contraction=1e-4, decrease=0.5, expansion=0.9, increase=2.0)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        assert step > 0
        assert np.isfinite(f_step)

    def test_goldstein_bounds_satisfied(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Goldstein(step=0.5, contraction=0.1, decrease=0.5, expansion=0.9, increase=1.5)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        grad = simple_quadratic.dphi
        slope = -float(np.real(np.vdot(grad.ravel(), grad.ravel())))
        # With slope < 0: upper_bound = phi + expansion*step*slope, lower_bound = phi + contraction*step*slope; we need upper <= f_step <= lower
        upper_bound = simple_quadratic.phi + step * ls.expansion * slope
        lower_bound = simple_quadratic.phi + step * ls.contraction * slope
        assert upper_bound <= f_step <= lower_bound
