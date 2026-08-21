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


def _effective_n(dataset) -> float:
    """Kish effective sample size from dataset weights; 1.0 if not available."""
    if dataset is None:
        return 1.0
    n_eff = getattr(dataset, "effective_n", None)
    if n_eff is None:
        return 1.0
    return float(n_eff)


@dataclass(init=True, repr=True)
class ChiSquared(Fi):
    """
    Chi-squared data fidelity term: (1/2) * sum(w * |residual|^2) / n_eff.

    Normalized by Kish effective sample size n_eff = (sum w)^2 / sum(w^2) so the
    data term is O(1) and L1 regularization lambda can be O(1). Uses the
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
        Evaluate chi-squared: (1/2) * sum(w * |residual|^2) / n_eff.

        n_eff is Kish effective sample size from dataset.effective_n. Forward is
        unweighted; weights w are applied only to the squared residuals.

        Args:
            x: Input array (Faraday depth or coefficients)

        Returns:
            Chi-squared value (scalar), normalized by n_eff
        """
        op = self.measurement_operator
        model_data = op.forward(x)  # unweighted forward
        op.dataset.model_data = model_data
        res = op.dataset.residual  # data - model_data
        chi_squared_vector = op.dataset.w * (res.real**2 + res.imag**2)
        xp = math_module(chi_squared_vector)
        raw = 0.5 * xp.sum(chi_squared_vector)
        n_eff = _effective_n(op.dataset)
        result = raw / n_eff
        self._func_value = result
        return result

    def calculate_gradient(self, x):
        """
        Calculate gradient of normalized chi-squared: (1/n_eff) * (-backward(weighted residual)).

        F(x) = (1/2) sum(w * |residual|^2) / n_eff, so dF/dx = (1/n_eff) * (-A^H(w*r)).

        Args:
            x: Input array (Faraday depth or coefficients)

        Returns:
            Gradient array (same shape as x), normalized by n_eff
        """
        op = self.measurement_operator
        model_data = op.forward(x)  # unweighted
        op.dataset.model_data = model_data
        weighted_res = op.dataset.w * op.dataset.residual  # w * (data - model_data)
        grad_raw = -op.backward(weighted_res)  # -A^H(weighted_res)
        n_eff = _effective_n(op.dataset)
        result = grad_raw / n_eff
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
