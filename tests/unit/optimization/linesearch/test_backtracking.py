"""Unit tests for BacktrackingArmijo line search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import BacktrackingArmijo


class TestBacktrackingArmijo:
    """Test Armijo backtracking: f(x+alpha*d) <= f(x) + c1*alpha*grad'*d."""

    def test_backtracking_returns_valid_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = BacktrackingArmijo(step=2.0, contraction=1e-4, decrease=0.5)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        assert step > 0
        assert np.isfinite(f_step)
        # For our quadratic, Armijo should accept step and we expect decrease
        assert f_step <= simple_quadratic.phi + 1e-3

    def test_backtracking_satisfies_armijo(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = BacktrackingArmijo(step=1.5, contraction=1e-4, decrease=0.5)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        grad = simple_quadratic.dphi
        m = -ls.contraction * float(np.real(np.vdot(grad.ravel(), grad.ravel())))
        assert f_step <= simple_quadratic.phi + step * m
