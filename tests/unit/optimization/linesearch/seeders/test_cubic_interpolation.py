"""Unit tests for CubicInterpolationSeeder."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.seeders import CubicInterpolationSeeder


class TestCubicInterpolationSeeder:
    """Test cubic/quadratic interpolation step seeder."""

    def test_first_call_returns_init_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        simple_quadratic.phi = 2.5
        seeder = CubicInterpolationSeeder(init_step=1.0)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step
        assert step == pytest.approx(1.0)

    def test_second_call_with_prev_step_uses_interpolation(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        simple_quadratic.phi = 2.5  # f(0) = 0.5*5 = 2.5
        seeder = CubicInterpolationSeeder(init_step=1.0)
        seeder.estimate_step_size(param_x, simple_quadratic)
        seeder.set_previous_step(0.5)
        # f at alpha=0.5: 0.5*(0.5)^2*5 = 0.625
        seeder._prev_f = 0.625
        simple_quadratic.phi = 2.5
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step >= seeder.min_step
        assert np.isfinite(step)

    def test_find_cubic_minimum_quadratic_fit(self):
        seeder = CubicInterpolationSeeder()
        # f0=2.5, g0=-5 (for ||x||^2=5, grad=x, g0=-||dphi||^2=-5), alpha_prev=0.5, f_prev=0.625
        alpha = seeder._find_cubic_minimum(2.5, -5.0, 0.5, 0.625)
        assert alpha is not None
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_set_previous_step(self):
        seeder = CubicInterpolationSeeder()
        seeder.set_previous_step(0.7)
        assert seeder._prev_step == 0.7
