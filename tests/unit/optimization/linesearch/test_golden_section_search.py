"""Unit tests for golden section search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import GoldenSection, pure_golden_section_search
from csromer.optimization.linesearch.golden_section_search import INV_GOLD


class TestPureGoldenSectionSearch:
    """Test pure_golden_section_search on 1D function."""

    def test_pure_golden_section_finds_minimum(self):
        # f(t) = (t - 1)^2, min at t=1
        f = lambda t: (t - 1.0) ** 2
        f_min, x_min = pure_golden_section_search(f, 0.0, 3.0, max_iter=50, tol=1e-7)
        assert abs(x_min - 1.0) < 0.01
        assert f_min < 0.01

    def test_pure_golden_section_returns_tuple(self):
        f = lambda t: t ** 2
        f_min, x_min = pure_golden_section_search(f, -1.0, 1.0, max_iter=20, tol=1e-6)
        assert isinstance(f_min, (float, np.floating))
        assert isinstance(x_min, (float, np.floating))


class TestGoldenSection:
    """Test GoldenSection line searcher."""

    def test_golden_section_returns_f_min_and_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = GoldenSection(tol=1e-6, max_iter=80)
        ls.objective_function = simple_quadratic
        f_min, x_min = ls.search(param_x)
        assert np.isfinite(f_min)
        assert np.isfinite(x_min)

    def test_inv_gold_constant(self):
        assert 0 < INV_GOLD < 1
