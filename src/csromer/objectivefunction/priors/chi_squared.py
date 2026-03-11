"""
Chi-squared data fidelity term for Faraday depth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...utils.array_utils import math_module
from ..fi import Fi

if TYPE_CHECKING:
    from ...transformers.measurement_operator import MeasurementOperator


@dataclass(init=True, repr=True)
class ChiSquared(Fi):
    """
    Chi-squared data fidelity term: (1/2) * sum(w * |residual|^2).

    Differentiable term that measures the fit between model and data. Uses the
    measurement operator's forward to compute residuals, and backward for gradient.
    No proximal operator (is_differentiable=True).

    Attributes:
        measurement_operator: Measurement operator (forward/backward)
        is_differentiable: Always True (chi-squared is differentiable)
    """
    measurement_operator: "MeasurementOperator" = None
    is_differentiable: bool = True

    def __post_init__(self):
        """
        Post-initialization: set wavelet transform on operator if both are provided.
        """
        super().__post_init__()
        if self.wavelet is not None and self.measurement_operator is not None:
            self.measurement_operator.wavelet_transform = self.wavelet

    def evaluate(self, x):
        """
        Evaluate chi-squared: (1/2) * sum(w * |residual|^2).

        Public method. Forward is unweighted: model_data = A(x). Residual
        residual = data - model_data. Weights w are applied only to the
        squared residuals (not to the forward operator).

        Args:
            x: Input array (Faraday depth or coefficients)

        Returns:
            Chi-squared value (scalar)
        """
        op = self.measurement_operator
        model_data = op.forward(x)  # unweighted forward
        op.dataset.model_data = model_data
        res = op.dataset.residual  # data - model_data
        chi_squared_vector = op.dataset.w * (res.real**2 + res.imag**2)
        xp = math_module(chi_squared_vector)
        result = 0.5 * xp.sum(chi_squared_vector)
        self._func_value = result
        return result

    def calculate_gradient(self, x):
        """
        Calculate gradient: -backward(weighted residual).

        F(x) = (1/2) sum(w * |residual|^2), residual = data - model_data.
        Gradient dF/dx = -A^H (w * residual). We pass w*residual to backward
        (adjoint); backward does not apply weights again. Sign: steepest
        descent updates x -= alpha*grad, so we return -A^H(w*r).

        Args:
            x: Input array (Faraday depth or coefficients)

        Returns:
            Gradient array (same shape as x)
        """
        op = self.measurement_operator
        model_data = op.forward(x)  # unweighted
        op.dataset.model_data = model_data
        weighted_res = op.dataset.w * op.dataset.residual  # w * (data - model_data)
        result = -op.backward(weighted_res)  # -A^H(weighted_res); backward = raw adjoint
        self._grad_value = result
        return result

    def calculate_prox(self, x, nu=0):
        """
        Proximal operator (not defined for chi-squared).

        Public method. Chi-squared is differentiable, so proximal is not needed.

        Args:
            x: Input array (unused)
            nu: Step size (unused)

        Raises:
            NotImplementedError: Always raised (chi-squared is differentiable)
        """
        raise NotImplementedError("ChiSquared is differentiable; proximal is not defined.")
