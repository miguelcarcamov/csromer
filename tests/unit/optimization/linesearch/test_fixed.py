"""Unit tests for Fixed step line search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import Fixed


class TestFixed:
    """Test Fixed line search returns (f(step), step)."""

    def test_fixed_returns_f_step_and_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Fixed(step=1.0)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        assert step == 1.0
        # f(1)=0 for our quadratic
        assert f_step == pytest.approx(0.0, abs=1e-10)

    def test_fixed_uses_step_from_kwargs(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Fixed(step=0.5)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x, step=0.25)
        assert step == 0.25
        # f(0.25)=0.5*(0.75)^2*5
        assert f_step == pytest.approx(0.5 * (0.75**2) * 5, rel=1e-8)
