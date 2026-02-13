"""
Chi-squared data fidelity term for Faraday depth reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..fi import Fi
from ...utils.array_utils import math_module

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
        
        Public method. Computes model data via forward operator, sets residual,
        and returns weighted sum of squared residuals.
        
        Args:
            x: Input array (Faraday depth or coefficients)
            
        Returns:
            Chi-squared value (scalar)
        """
        op = self.measurement_operator
        model_data = op.forward(x)
        op.dataset.model_data = model_data
        res = op.dataset.residual
        chi_squared_vector = op.dataset.w * (res.real**2 + res.imag**2)
        xp = math_module(chi_squared_vector)
        result = 0.5 * xp.sum(chi_squared_vector)
        self._func_value = result
        return result

    def calculate_gradient(self, x):
        """
        Calculate gradient: backward(weighted residual).
        
        Public method. Computes model data, sets residual, weights it, and applies
        backward operator (adjoint of forward).
        
        Args:
            x: Input array (Faraday depth or coefficients)
            
        Returns:
            Gradient array (same shape as x)
        """
        op = self.measurement_operator
        model_data = op.forward(x)
        op.dataset.model_data = model_data
        weighted_res = op.dataset.w * op.dataset.residual
        result = op.backward(weighted_res)
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
