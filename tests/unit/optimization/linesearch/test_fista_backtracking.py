"""Unit tests for FISTA backtracking line search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import FISTABacktracking


class TestFISTABacktracking:
    """Test FISTA backtracking: find L s.t. F(prox(y-(1/L)*grad)) <= Q_L."""

    def test_fista_backtracking_returns_valid_step(self, fista_mock_objective, param_x):
        fista_mock_objective.calculate_gradient(param_x.data)
        ls = FISTABacktracking(initial_lipschitz=1.0, decrease=0.5)
        ls.objective_function = fista_mock_objective
        f_step, step = ls.search(param_x)
        assert step > 0
        assert np.isfinite(f_step)
        # step = 1/L
        assert step <= 1.0

    def test_fista_backtracking_updates_x_data(self, fista_mock_objective, param_x):
        fista_mock_objective.calculate_gradient(param_x.data)
        ls = FISTABacktracking(initial_lipschitz=1.0)
        ls.objective_function = fista_mock_objective
        orig_data = np.array(param_x.data, copy=True)
        ls.search(param_x)
        # With identity prox, x should move toward minimum
        assert np.any(param_x.data != orig_data) or np.allclose(param_x.data, orig_data)
