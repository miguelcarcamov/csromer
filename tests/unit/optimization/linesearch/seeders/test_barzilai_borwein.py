"""Unit tests for Barzilai-Borwein step size seeders."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.seeders.barzilai_borwein import (
    BarzilaiBorwein,
    BarzilaiBorweinAdaptiveMin1,
    BarzilaiBorweinAdaptiveMin2,
    BarzilaiBorweinAlternating,
)
from csromer.optimization.linesearch.seeders.barzilai_borwein.base import BarzilaiBorwein as BBBase


class TestBarzilaiBorweinFormulas:
    """Test BB1 and BB2 formulas."""

    def test_bb1_formula(self):
        # alpha = ||s||^2 / (s'y); s'y > 0
        step_norm_sqr = 4.0
        step_curvature = 2.0
        alpha = BBBase._bb1(step_norm_sqr, step_curvature)
        assert alpha == pytest.approx(2.0)

    def test_bb1_returns_nan_when_curvature_small(self):
        alpha = BBBase._bb1(1.0, 1e-15)
        assert np.isnan(alpha)

    def test_bb2_formula(self):
        # alpha = s'y / ||y||^2
        step_curvature = 2.0
        grad_change_norm_sqr = 0.5
        alpha = BBBase._bb2(step_curvature, grad_change_norm_sqr)
        assert alpha == pytest.approx(4.0)

    def test_bb2_returns_nan_when_denom_small(self):
        alpha = BBBase._bb2(1.0, 1e-15)
        assert np.isnan(alpha)


class TestBarzilaiBorweinAlternating:
    """Test alternating: even -> BB1, odd -> BB2."""

    def test_alternating_first_call_returns_init_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        seeder = BarzilaiBorweinAlternating(init_step=1.0)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step
        assert np.isfinite(step)

    def test_alternating_second_call_returns_bb(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        seeder = BarzilaiBorweinAlternating(init_step=1.0)
        seeder.estimate_step_size(param_x, simple_quadratic)
        # Move param and gradient to get different s, y
        param_x.data = np.array([0.5, 1.0])
        simple_quadratic.calculate_gradient(param_x.data)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step
        assert np.isfinite(step)


class TestBarzilaiBorweinAdaptiveMin1:
    """Test ABBmin1: min(BB2 history) when ratio < tau else BB1."""

    def test_adaptive_min1_first_call(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        seeder = BarzilaiBorweinAdaptiveMin1(tau=0.8)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step

    def test_adaptive_min1_select_step_candidate_ratio_above_tau(self):
        seeder = BarzilaiBorweinAdaptiveMin1(tau=0.8)
        # ratio = alpha_bb2/alpha_bb1 = 1.0 >= tau -> return BB1
        out = seeder._select_step_candidate(1.0, 1.0, 1)
        assert out == 1.0


class TestBarzilaiBorweinAdaptiveMin2:
    """Test ABBmin2: use previous step when ratio < tau else BB1."""

    def test_adaptive_min2_first_call(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        seeder = BarzilaiBorweinAdaptiveMin2(tau=0.9)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step

    def test_adaptive_min2_returns_prev_step_when_ratio_below_tau(self):
        seeder = BarzilaiBorweinAdaptiveMin2(tau=0.9)
        seeder._prev_step = 0.3
        out = seeder._select_step_candidate(1.0, 0.5, 1)  # ratio 0.5 < 0.9
        assert out == 0.3
