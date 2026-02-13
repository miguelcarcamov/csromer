#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective function as sum of terms (Fi instances). Supports both legacy API
and Pyralysis-style filtering (differentiable_only / nondifferentiable_only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union, Any

import numpy as np

try:
    import dask.array as da
except ImportError:
    da = None

from ..utils.array_utils import maybe_compute, zeros_like
from .fi import Fi


@dataclass(init=True, repr=True)
class OFunction:
    """
    Objective function F = sum_i penalization_factor_i * term_i(x).
    Supports filtering by differentiable / non-differentiable terms for FISTA-style methods.
    Persists last objective (phi) and gradient (dphi) like Pyralysis.
    """
    F: Optional[List[Fi]] = None
    persist_gradient: bool = False
    values: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    nfuncs: Optional[int] = field(init=False, default=None)
    prox_functions: List[Fi] = field(init=False, default_factory=list)
    phi: float = field(init=False, default=0.0)
    dphi: Optional[Any] = field(init=False, default=None)

    def __post_init__(self):
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

    def getProxFunctions(self):
        return self.prox_functions

    def getValues(self):
        return self.values

    def getLambda(self, _id=0):
        return self.F[_id].reg

    def setLambda(self, reg=0.0, _id=0):
        self.F[_id].reg = reg

    def evaluate(self, x):
        """Legacy: full objective value at x. Persists phi and each term's _func_value (lazy when dask)."""
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
        Gradient at x. If differentiable_only is True, only differentiable terms are included.
        Sets self.dphi (and each term's _grad_value) and returns it. Keeps dask when x is dask.
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
            out[:] = maybe_compute(res)
        return res

    def calc_prox(self, x, nu=0, _id=0):
        """Legacy: proximal step (single term or composition)."""
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
        """List of objective function terms (same as F)."""
        return getattr(self, "F", [])

    def terms_parameter(self, parameter):
        """Set parameter on every term."""
        for term in self.F:
            term.parameter = parameter
        self._parameter = parameter

    def terms_penalization(self, penalization: Union[float, list, np.ndarray]):
        """Set penalization factor (reg) on every term. If list/array, length must match terms."""
        if np.isscalar(penalization):
            for term in self.F:
                term.reg = float(penalization)
        else:
            if len(penalization) != len(self.F):
                raise ValueError("penalization length must match number of terms")
            for term, pen in zip(self.F, penalization):
                term.reg = float(pen)

    def _nondiff_terms(self) -> List[Fi]:
        """Terms with is_differentiable=False (for FISTA proximal step)."""
        return [t for t in self.F if not getattr(t, "is_differentiable", True)]

    def apply_prox_nondiff(self, x, nu=0):
        """Apply proximal of all non-differentiable terms in sequence (FISTA: expect one)."""
        out = x
        for term in self._nondiff_terms():
            out = term.calculate_prox(out, nu)
        return out

    def get_lambda_nondiff(self):
        """Reg of first non-differentiable term, or None if none."""
        nondiff = self._nondiff_terms()
        return float(nondiff[0].reg) if nondiff else None

    def set_lambda_nondiff(self, reg: float):
        """Set reg of first non-differentiable term (e.g. for FISTA cooling)."""
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
    ):
        """
        Evaluate the objective (or only differentiable / only non-differentiable terms).
        Persists phi (lazy when dask). Returns computed float for callers that need a scalar.
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
        return float(maybe_compute(value))

