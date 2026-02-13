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
    Chi-squared data term: (1/2) * sum(w * |residual|^2).

    Differentiable: evaluation uses the measurement operator's forward;
    gradient uses backward(weighted residual). No proximal (is_differentiable=True).
    """
    measurement_operator: "MeasurementOperator" = None
    is_differentiable: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.wavelet is not None and self.measurement_operator is not None:
            self.measurement_operator.wavelet_transform = self.wavelet

    def evaluate(self, x):
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
        op = self.measurement_operator
        model_data = op.forward(x)
        op.dataset.model_data = model_data
        weighted_res = op.dataset.w * op.dataset.residual
        result = op.backward(weighted_res)
        self._grad_value = result
        return result

    def calculate_prox(self, x, nu=0):
        raise NotImplementedError("ChiSquared is differentiable; proximal is not defined.")
