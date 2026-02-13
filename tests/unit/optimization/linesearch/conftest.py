"""Fixtures for line search and seeder tests: mock objective, Parameter."""
import numpy as np
import pytest

from csromer.reconstruction.parameter import Parameter


class SimpleQuadraticObjective:
    """F(x) = 0.5 * ||x||^2, grad = x. f(alpha) = F(x - alpha*grad) = 0.5*(1-alpha)^2||x||^2, min at alpha=1."""

    def __init__(self):
        self.phi = 0.0
        self.dphi = None

    def evaluate(self, x):
        x = np.asarray(x)
        return float(0.5 * np.real(np.vdot(x.ravel(), x.ravel())))

    def calculate_gradient(self, x):
        self.dphi = np.array(x, copy=True)
        self.phi = self.evaluate(x)

    def calculate_function(self, x, differentiable_only=False, nondifferentiable_only=False):
        if nondifferentiable_only:
            return 0.0
        return self.evaluate(x)


@pytest.fixture
def simple_quadratic():
    """Objective F(x)=0.5||x||^2, grad=x. Minimum along -grad at alpha=1."""
    return SimpleQuadraticObjective()


@pytest.fixture
def param_x():
    """Parameter with data [1.0, 2.0]; ||x||^2=5, min at alpha=1."""
    return Parameter(data=np.array([1.0, 2.0], dtype=np.float64))


@pytest.fixture
def param_small():
    """Parameter with small data for bracketing tests."""
    return Parameter(data=np.array([0.1, 0.2], dtype=np.float64))


class FISTAMockObjective:
    """F = f + g, f smooth, g prox. For FISTA backtracking: f(x)=0.5||x||^2, g=0, prox=id."""

    def __init__(self):
        self.phi = 0.0
        self.dphi = None
        self.F = [self._smooth_term()]

    class _SmoothTerm:
        is_differentiable = True

        def evaluate(self, x):
            x = np.asarray(x)
            return float(0.5 * np.real(np.vdot(x.ravel(), x.ravel())))

    def _smooth_term(self):
        return self._SmoothTerm()

    def evaluate(self, x):
        x = np.asarray(x)
        return float(0.5 * np.real(np.vdot(x.ravel(), x.ravel())))

    def calculate_gradient(self, x):
        self.dphi = np.array(x, copy=True)
        self.phi = self.evaluate(x)

    def calculate_function(self, x, differentiable_only=False, nondifferentiable_only=False):
        if nondifferentiable_only:
            return 0.0
        return self.evaluate(x)


@pytest.fixture
def fista_mock_objective():
    """Objective with F = [smooth term], no non-diff term; prox = identity."""
    return FISTAMockObjective()
