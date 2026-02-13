"""Unit tests for Brent line search (golden section + parabolic)."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import Brent


class TestBrent:
    """Test Brent's method finds minimum in bracket."""

    def test_brent_returns_f_min_and_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Brent(tol=1e-8, max_iter=100)
        ls.objective_function = simple_quadratic
        f_min, x_min = ls.search(param_x)
        assert np.isfinite(f_min)
        assert np.isfinite(x_min)
        assert x_min >= 0

    def test_brent_finds_near_optimal_for_quadratic(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Brent(tol=1e-6, max_iter=200)
        ls.objective_function = simple_quadratic
        f_min, x_min = ls.search(param_x)
        # True minimum at alpha=1 for f(alpha)=0.5*(1-alpha)^2*5
        assert abs(x_min - 1.0) < 0.1
        assert f_min < 0.1
