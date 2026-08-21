"""Unit tests for StepSizeSeeder base."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.seeders.base import StepSizeSeeder
from csromer.reconstruction.parameter import Parameter


class ConcreteSeeder(StepSizeSeeder):

    def estimate_step_size(self, x, objective_function):
        return self.init_step


class TestStepSizeSeeder:
    """Test StepSizeSeeder base and concrete subclass."""

    def test_seeder_abstract(self):
        with pytest.raises(TypeError):
            StepSizeSeeder()

    def test_concrete_seeder_returns_init_step(self, simple_quadratic, param_x):
        seeder = ConcreteSeeder(init_step=0.7)
        step = seeder.estimate_step_size(param_x, simple_quadratic)
        assert step == 0.7

    def test_seeder_min_step_init_step_defaults(self):
        seeder = ConcreteSeeder()
        assert seeder.min_step > 0
        assert seeder.init_step == 1.0
