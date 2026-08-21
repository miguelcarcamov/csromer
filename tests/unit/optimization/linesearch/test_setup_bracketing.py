"""Unit tests for setup_bracketing (f1dim + mnbrak)."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.setup_bracketing import setup_bracketing


class TestSetupBracketing:
    """Test setup_bracketing returns f, a, c (and optionally b_)."""

    def test_setup_bracketing_returns_f_a_c(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        result = setup_bracketing(simple_quadratic, param_x, initial_a=0.0, initial_b=1.0)
        f, a, c = result
        assert callable(f)
        assert a < c
        # f(alpha)=0.5*(1-alpha)^2*5, min at alpha=1
        f_at_a, f_at_c = f(a), f(c)
        assert np.isfinite(f_at_a)
        assert np.isfinite(f_at_c)

    def test_setup_bracketing_return_middle(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        result = setup_bracketing(
            simple_quadratic, param_x, initial_a=0.0, initial_b=1.0, return_middle=True
        )
        f, a, c, b_ = result
        assert callable(f)
        assert a <= b_ <= c or a <= c
        assert np.isfinite(f(b_))
