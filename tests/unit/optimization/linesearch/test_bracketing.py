"""Unit tests for mnbrak bracketing."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.bracketing import GOLD, TINY, mnbrak


class TestMnbrak:
    """Test bracketing triplet ax < bx < cx with f(bx) < f(ax), f(bx) < f(cx)."""

    def test_mnbrak_quadratic(self):
        # f(t) = (t - 1)^2, minimum at t=1; use ax, bx so minimum is between them
        f = lambda t: (t - 1.0) ** 2
        ax, bx, cx = 0.0, 1.5, None  # min at 1 between 0 and 1.5
        a, b, c, fa, fb, fc = mnbrak(f, ax, bx, cx)
        assert a < b < c
        assert fb <= fa
        assert fb <= fc

    def test_mnbrak_with_cx(self):
        f = lambda t: (t - 2.0) ** 2
        ax, bx, cx = 0.0, 1.0, 3.0
        a, b, c, fa, fb, fc = mnbrak(f, ax, bx, cx)
        assert a < b < c
        assert fb <= fa
        assert fb <= fc

    def test_mnbrak_constants(self):
        assert GOLD > 1.0
        assert TINY > 0 and TINY < 1e-10
