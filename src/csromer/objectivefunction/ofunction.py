#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective function as sum of terms (Fi instances). Supports both legacy API
and Pyralysis-style filtering (differentiable_only / nondifferentiable_only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np

try:
    import dask.array as da
except ImportError:
    da = None

from ..utils.array_utils import zeros_like
from .fi import Fi


@dataclass(init=True, repr=True)
class OFunction:
    """
    Objective function F = sum_i penalization_factor_i * term_i(x).

    Manages a list of objective function terms (Fi instances). Supports filtering
    by differentiable / non-differentiable terms for FISTA-style methods. Persists
    last objective (phi) and gradient (dphi) like Pyralysis.

    Attributes:
        F: List of objective function terms (Fi instances)
        persist_gradient: If True, persist gradient in dask (default: False)
        values: Per-term objective values (from last evaluate)
        nfuncs: Number of terms
        prox_functions: List of terms with proximal operators (same as F)
        phi: Last computed objective value
        dphi: Last computed gradient
    """
    F: Optional[List[Fi]] = None
    persist_gradient: bool = False
    values: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    nfuncs: Optional[int] = field(init=False, default=None)
    prox_functions: List[Fi] = field(init=False, default_factory=list)
    phi: float = field(init=False, default=0.0)
    dphi: Optional[Any] = field(init=False, default=None)

    def __post_init__(self):
        """
        Post-initialization: set up terms and initialize arrays.
        """
        if self.F is None:
            self.F = []
            self.values = np.array([])
            self.nfuncs = None
            self.prox_functions = []
        else:
            self.values = np.zeros(len(self.F))
            self.nfuncs = len(self.F)
            self.prox_functions = [f_i for f_i in self.F]
        self.phi = 0.0
        self.dphi = None

    def getProxFunctions(self) -> List[Fi]:
        """
        Legacy API: get list of terms with proximal operators.

        Returns:
            List of Fi instances (same as F)
        """
        return self.prox_functions

    def getValues(self) -> np.ndarray:
        """
        Legacy API: get per-term objective values from last evaluate.

        Returns:
            Array of term values
        """
        return self.values

    def getLambda(self, _id: int = 0) -> float:
        """
        Legacy API: get regularization factor for term _id.

        Args:
            _id: Term index (default: 0)

        Returns:
            Regularization factor (reg)
        """
        return self.F[_id].reg

    def setLambda(self, reg: float = 0.0, _id: int = 0):
        """
        Legacy API: set regularization factor for term _id.

        Args:
            reg: Regularization factor
            _id: Term index (default: 0)
        """
        self.F[_id].reg = reg

    def evaluate(self, x) -> float:
        """
        Legacy API: evaluate full objective value at x.

        Persists phi and each term's _func_value (lazy when dask).

        Args:
            x: Input array (Faraday depth or coefficients)

        Returns:
            Objective value (float)
        """
        ret = 0.0
        for i in range(0, len(self.F)):
            self.values[i] = self.F[i].evaluate(x)
            self.F[i]._func_value = self.values[i]
            ret += self.F[i].reg * self.values[i]
        self.phi = ret
        return ret

    def calculate_gradient(
        self,
        x,
        *,
        iteration: int = 0,
        out=None,
        mask=None,
        differentiable_only: bool = False,
    ):
        """
        Calculate gradient at x.

        Public method. If differentiable_only is True, only differentiable terms
        are included. Sets self.dphi (and each term's _grad_value) and returns it.
        Keeps dask when x is dask.

        Args:
            x: Input array (Faraday depth or coefficients)
            iteration: Iteration number (passed to terms)
            out: Optional output array (will be filled with computed gradient)
            mask: Optional mask (not currently used)
            differentiable_only: If True, skip non-differentiable terms

        Returns:
            Gradient array (same shape/type as x)
        """
        res = zeros_like(x)
        for f_i in self.F:
            if differentiable_only and not getattr(f_i, "is_differentiable", True):
                continue
            f_i.iteration = iteration
            g = f_i.calculate_gradient(x)
            f_i._grad_value = g
            res = res + f_i.reg * g
        if self.persist_gradient and da is not None and isinstance(res, da.Array):
            res = res.persist()
        self.dphi = res
        if out is not None:
            out[:] = np.asarray(res.compute()) if hasattr(res, "compute") else np.asarray(res)
        return res

    def calc_prox(self, x, nu: float = 0, _id: int = 0):
        """
        Legacy API: proximal step (single term or composition).

        Args:
            x: Input array
            nu: Step size parameter
            _id: Term index (if single term)

        Returns:
            Proximal result
        """
        if len(self.prox_functions) == 1:
            f_i = self.F[_id]
            proximal = f_i.calculate_prox(x, nu)
        else:
            proximal = x
            for i in range(len(self.prox_functions)):
                proximal = self.F[i].calculate_prox(proximal, nu)
        return proximal

    # --- Pyralysis-style API ---

    @property
    def terms(self) -> List[Fi]:
        """
        List of objective function terms (same as F).

        Public property for Pyralysis-style API.
        """
        return getattr(self, "F", [])

    def terms_parameter(self, parameter):
        """
        Set parameter on every term.

        Public method for Pyralysis-style API.

        Args:
            parameter: Parameter object to set on all terms
        """
        for term in self.F:
            term.parameter = parameter
        self._parameter = parameter

    def terms_penalization(self, penalization: Union[float, list, np.ndarray]):
        """
        Set penalization factor (reg) on every term.

        Public method for Pyralysis-style API. If list/array, length must match terms.

        Args:
            penalization: Penalization factor(s). Scalar for all terms, or list/array per term.

        Raises:
            ValueError: If array length doesn't match number of terms
        """
        if np.isscalar(penalization):
            for term in self.F:
                term.reg = float(penalization)
        else:
            if len(penalization) != len(self.F):
                raise ValueError("penalization length must match number of terms")
            for term, pen in zip(self.F, penalization):
                term.reg = float(pen)

    def _nondiff_terms(self) -> List[Fi]:
        """
        Get terms with is_differentiable=False (for FISTA proximal step).

        Private method: used internally by FISTA-style methods.

        Returns:
            List of non-differentiable terms
        """
        return [t for t in self.F if not getattr(t, "is_differentiable", True)]

    def apply_prox_nondiff(self, x, nu: float = 0):
        """
        Apply proximal of all non-differentiable terms in sequence.

        Public method for FISTA-style optimizers. Expects one non-differentiable term
        (typically L1 or TV).

        Args:
            x: Input array
            nu: Step size parameter

        Returns:
            Result after applying all non-differentiable proximals
        """
        out = x
        for term in self._nondiff_terms():
            out = term.calculate_prox(out, nu)
        return out

    def get_lambda_nondiff(self) -> Optional[float]:
        """
        Get reg of first non-differentiable term.

        Public method for FISTA-style optimizers.

        Returns:
            Regularization factor or None if no non-differentiable terms
        """
        nondiff = self._nondiff_terms()
        return float(nondiff[0].reg) if nondiff else None

    def set_lambda_nondiff(self, reg: float):
        """
        Set reg of first non-differentiable term (e.g. for FISTA cooling).

        Public method for FISTA-style optimizers.

        Args:
            reg: Regularization factor
        """
        nondiff = self._nondiff_terms()
        if nondiff:
            nondiff[0].reg = float(reg)

    def calculate_function(
        self,
        x,
        *,
        mask=None,
        differentiable_only: bool = False,
        nondifferentiable_only: bool = False,
    ) -> float:
        """
        Evaluate the objective (or only differentiable / only non-differentiable terms).

        Public method for Pyralysis-style API. Persists phi (lazy when dask).
        Returns computed float for callers that need a scalar.

        Args:
            x: Input array (Faraday depth or coefficients)
            mask: Optional mask (not currently used)
            differentiable_only: If True, only evaluate differentiable terms
            nondifferentiable_only: If True, only evaluate non-differentiable terms

        Returns:
            Objective value (float)
        """
        value = 0.0
        for term in self.F:
            if differentiable_only and not getattr(term, "is_differentiable", True):
                continue
            if nondifferentiable_only and getattr(term, "is_differentiable", True):
                continue
            v = term.evaluate(x)
            term._func_value = v
            value += term.reg * v
        self.phi = value
        return float(value.compute()) if hasattr(value, "compute") else float(np.asarray(value).item())
