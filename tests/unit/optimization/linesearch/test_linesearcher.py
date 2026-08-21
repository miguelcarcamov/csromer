"""Unit tests for LineSearcher base: _get_initial_step_size, _read_kwargs."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import BacktrackingArmijo, Fixed
from csromer.optimization.linesearch.seeders import StepSizeSeeder


class DummySeeder(StepSizeSeeder):

    def estimate_step_size(self, x, objective_function):
        return 0.5


class TestLineSearcherBase:
    """Test LineSearcher _get_initial_step_size and _read_kwargs via concrete subclass."""

    def test_get_initial_step_size_no_seeder(self, simple_quadratic, param_x):
        ls = Fixed(step=2.0)
        ls.objective_function = simple_quadratic
        step = ls._get_initial_step_size(param_x)
        assert step == 2.0

    def test_get_initial_step_size_with_seeder(self, simple_quadratic, param_x):
        ls = Fixed(step=1.0, seeder=DummySeeder())
        ls.objective_function = simple_quadratic
        step = ls._get_initial_step_size(param_x)
        assert step == 0.5

    def test_read_kwargs_step_tol_max_iter(self):
        ls = BacktrackingArmijo()
        ls._read_kwargs(step=3.0, tol=1e-5, max_iter=50)
        assert ls.step == 3.0
        assert ls.tol == 1e-5
        assert ls.max_iter == 50

    def test_read_kwargs_seeder(self, simple_quadratic, param_x):
        ls = Fixed(seeder=None)
        ls.objective_function = simple_quadratic
        ls._read_kwargs(seeder=DummySeeder())
        assert ls.seeder is not None
        assert ls._get_initial_step_size(param_x) == 0.5
