"""Unit tests for GLL (Grippo-Lampariello-Lucidi) Armijo line search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import GLLArmijo


class TestGLLArmijo:
    """Test GLL: f(x+alpha*d) <= max_{j in window} f(x_{k-j}) + c1*alpha*grad'*d."""

    def test_gll_returns_valid_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = GLLArmijo(step=1.0, memory_window=5)
        ls.objective_function = simple_quadratic
        f_step, step = ls.search(param_x)
        assert step > 0
        assert np.isfinite(f_step)

    def test_gll_update_function_history(self):
        ls = GLLArmijo(memory_window=3)
        ls._update_function_history(1.0)
        ls._update_function_history(2.0)
        ls._update_function_history(0.5)
        ref = ls._get_reference_value()
        assert ref == 2.0

    def test_gll_reset_history(self):
        ls = GLLArmijo(memory_window=3)
        ls._update_function_history(1.0)
        ls.reset_history()
        assert ls._function_history is None
        assert ls._current_max == float("-inf")
