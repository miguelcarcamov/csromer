"""Unit tests for Fibonacci search."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch import Fibonacci, pure_fibonacci_search
from csromer.optimization.linesearch.fibonacci_search import _fib_sequence


class TestFibSequence:
    """Test _fib_sequence helper."""

    def test_fib_sequence_n_0(self):
        assert _fib_sequence(0) == [1]

    def test_fib_sequence_n_1(self):
        assert _fib_sequence(1) == [1, 1]

    def test_fib_sequence_n_5(self):
        fib = _fib_sequence(5)
        assert fib == [1, 1, 2, 3, 5, 8]


class TestPureFibonacciSearch:
    """Test pure_fibonacci_search on 1D function."""

    def test_pure_fibonacci_finds_minimum(self):
        f = lambda t: (t - 1.0) ** 2
        f_min, x_min = pure_fibonacci_search(f, 0.0, 3.0, max_iter=30, tol=1e-6)
        assert abs(x_min - 1.0) < 0.1
        assert f_min < 0.1

    def test_pure_fibonacci_narrow_interval(self):
        f = lambda t: t ** 2
        f_min, x_min = pure_fibonacci_search(f, -0.1, 0.1, max_iter=20, tol=1e-8)
        assert abs(x_min) < 0.1


class TestFibonacci:
    """Test Fibonacci line searcher."""

    def test_fibonacci_returns_f_min_and_step(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        ls = Fibonacci(tol=1e-6, max_iter=50)
        ls.objective_function = simple_quadratic
        f_min, x_min = ls.search(param_x)
        assert np.isfinite(f_min)
        assert np.isfinite(x_min)
