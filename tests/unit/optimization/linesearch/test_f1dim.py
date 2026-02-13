"""Unit tests for f1dim (1D line search function)."""
import numpy as np
import pytest

pytest.importorskip("pywt")

from csromer.optimization.linesearch.f1dim import f1dim


class TestF1dim:
    """Test f(alpha) = F(x - alpha * dphi)."""

    def test_f1dim_requires_dphi(self, simple_quadratic, param_x):
        simple_quadratic.dphi = None
        with pytest.raises(ValueError, match="dphi must be set"):
            f1dim(simple_quadratic, param_x)

    def test_f1dim_quadratic_min_at_one(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        f = f1dim(simple_quadratic, param_x)
        # F(x) = 0.5||x||^2, grad=x, f(alpha)=0.5*(1-alpha)^2||x||^2, min at alpha=1
        assert f(0.0) == pytest.approx(2.5, rel=1e-10)  # 0.5 * 5
        assert f(1.0) == pytest.approx(0.0, rel=1e-10)
        assert f(2.0) == pytest.approx(2.5, rel=1e-10)

    def test_f1dim_returns_scalar(self, simple_quadratic, param_x):
        simple_quadratic.calculate_gradient(param_x.data)
        f = f1dim(simple_quadratic, param_x)
        v = f(0.5)
        assert isinstance(v, (float, np.floating))
        assert np.isfinite(v)
