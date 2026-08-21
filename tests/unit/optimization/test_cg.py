"""Unit tests for non-linear Conjugate Gradient (Pyralysis-style variants)."""
import numpy as np
import pytest

pytest.importorskip("pywt")  # csromer.optimization chain pulls in dictionaries -> pywt

from csromer.optimization import (
    ConjugateGradient,
    DaiYuan,
    FletcherReeves,
    GradientNormError,
    HagerZhang,
    HestenesStiefel,
    PolakRibiere,
)
from csromer.reconstruction.parameter import Parameter


class TestConjugateGradientParameter:
    """Test beta formulas for each CG variant (gradient convention: g, not residual)."""

    def test_fletcher_reeves_beta(self):
        grad = np.ones((16, ))
        grad_prev = np.ones((16, ))
        dir_prev = np.ones((16, ))
        cg = FletcherReeves()
        beta, _, _ = cg.conjugate_gradient_parameter(grad, grad_prev, dir_prev)
        assert abs(beta - 1.0) < 1e-10

    def test_polak_ribiere_beta_same_grad(self):
        grad = np.ones((16, ))
        grad_prev = np.ones((16, ))
        dir_prev = np.ones((16, ))
        cg = PolakRibiere()
        beta, _, _ = cg.conjugate_gradient_parameter(grad, grad_prev, dir_prev)
        assert abs(beta - 0.0) < 1e-10

    def test_dai_yuan_beta(self):
        grad = np.full((16, ), 2.0)
        grad_prev = np.ones((16, ))
        dir_prev = np.ones((16, ))
        cg = DaiYuan()
        beta, _, _ = cg.conjugate_gradient_parameter(grad, grad_prev, dir_prev)
        # norm2_g=4*16=64, denom = dir_prev^T (grad - grad_prev) = 1*16 = 16, beta = 64/16 = 4
        assert abs(beta - 4.0) < 1e-10

    def test_hestenes_stiefel_beta(self):
        grad = np.full((16, ), 2.0)
        grad_prev = np.ones((16, ))
        dir_prev = np.ones((16, ))
        cg = HestenesStiefel()
        beta, _, _ = cg.conjugate_gradient_parameter(grad, grad_prev, dir_prev)
        # numer = g^T (g - g_prev) = 2*16 = 32, denom = d^T (g - g_prev) = 16, beta = 2
        assert abs(beta - 2.0) < 1e-10

    def test_gradient_norm_error(self):
        cg = DaiYuan()
        grad = np.ones((4, ))
        grad_prev = np.zeros((4, ))
        dir_prev = np.ones((4, ))
        with pytest.raises(GradientNormError):
            cg.conjugate_gradient_parameter(grad, grad_prev, dir_prev)


class TestConjugateGradientRestart:
    """Test Powell and negative-beta restart (Pyralysis-style)."""

    def test_should_restart_negative_beta(self):
        cg = FletcherReeves()
        assert cg._should_restart(-0.5, 0.0, 1.0) is True
        assert cg._should_restart(0.0, 0.0, 1.0) is True

    def test_should_restart_positive_beta_no_restart(self):
        cg = FletcherReeves()
        assert cg._should_restart(0.5, 256.0, 256.0) is False

    def test_should_restart_powell_condition(self):
        cg = FletcherReeves()
        assert cg._should_restart(0.5, -256.0, 256.0) is True


class TestConjugateGradientMethodNames:
    """Test method_name() for each variant."""

    @pytest.mark.parametrize(
        "variant_class,expected_name",
        [
            (FletcherReeves, "Fletcher-Reeves"),
            (PolakRibiere, "Polak-Ribiere-Polyak"),
            (HestenesStiefel, "Hestenes-Stiefel"),
            (DaiYuan, "Dai-Yuan"),
            (HagerZhang, "Hager-Zhang"),
        ],
    )
    def test_method_name(self, variant_class, expected_name):
        cg = variant_class()
        assert cg.method_name() == expected_name
