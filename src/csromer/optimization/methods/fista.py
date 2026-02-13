"""
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
Uses only F_obj (Pyralysis-style): gradient from differentiable terms, prox from
non-differentiable terms. Supports monotone FISTA (MFISTA) and adaptive restart.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ...utils.array_utils import maybe_compute
from ..optimizer import Optimizer


def _f_value(F, x) -> float:
    """Evaluate F(x) and return scalar (dask-safe)."""
    v = F(x)
    return float(maybe_compute(v)) if hasattr(v, "compute") else float(np.asarray(v).item())


def _inner_real(a, b) -> float:
    """Real part of inner product (dask-safe)."""
    out = np.real(np.vdot(np.ravel(a), np.ravel(b)))
    return float(maybe_compute(out))


@dataclass(init=True, repr=True)
class FISTA(Optimizer):
    """
    FISTA for objectives F(x) = f(x) + g(x) with smooth f and proximal for g.
    Uses F_obj only: gradient = F_obj.calculate_gradient(..., differentiable_only=True),
    prox = F_obj.apply_prox_nondiff(...). Optional: monotone (MFISTA), adaptive restart.
    """

    noise: float = None
    monotonic: bool = False
    adaptive_restart: Optional[Literal["function", "gradient"]] = None

    def run(self):
        # Gradient of smooth part
        def grad_f(z):
            return self.F_obj.calculate_gradient(z, differentiable_only=True)

        # Prox and lambda for non-diff term (cooling)
        get_lam = self.F_obj.get_lambda_nondiff
        set_lam = self.F_obj.set_lambda_nondiff

        max_iter = self.maxiter
        if max_iter is None and self.noise is not None and get_lam() is not None:
            lam = get_lam()
            if self.noise != 0.0 and not (isinstance(self.noise, float) and np.isnan(self.noise)):
                max_iter = int(max(1, np.floor(lam / self.noise)))
            else:
                max_iter = 500
            if self.verbose:
                print("Iterations set to " + str(max_iter))

        if max_iter is None:
            max_iter = 500

        ret, x = self._fista_algorithm(
            self.guess_param.data,
            self.F_obj.evaluate,
            grad_f,
            self.F_obj.apply_prox_nondiff,
            get_lam,
            set_lam,
            max_iter,
            self.tol,
            self.guess_param.n,
            self.noise,
            self.verbose,
            self.monotonic,
            self.adaptive_restart,
        )
        param = copy.deepcopy(self.guess_param)
        param.data = x
        return ret, param

    @staticmethod
    def _fista_algorithm(
        x=None,
        F=None,
        grad_f=None,
        prox_g=None,
        get_lambda=None,
        set_lambda=None,
        max_iter=500,
        tol=np.finfo(np.float32).tiny,
        n=None,
        noise=None,
        verbose=True,
        monotonic=False,
        adaptive_restart=None,
    ):
        if x is None and n is not None:
            x = np.zeros(n, dtype=np.complex64)
        x = np.array(x, copy=True)
        t = 1.0
        z = np.array(x, copy=True)

        if noise is None:
            noise = 1e-5

        lam = get_lambda() if get_lambda else None
        if lam is not None and noise >= lam:
            if verbose:
                print("Error, noise cannot be greater than lambda")
            return _f_value(F, x), x

        f_prev = _f_value(F, x)
        for it in range(0, max_iter):
            x_old = np.array(x, copy=True)
            y = np.array(z, copy=True)

            z = z - grad_f(z)
            x = prox_g(z)

            f_new = _f_value(F, x)
            progress = True

            if monotonic and f_new > f_prev:
                x = x_old
                z = x_old
                t = 1.0
                f_new = f_prev
                progress = False
                if verbose and it % 10 == 0:
                    print("Iteration: {} (monotone reject) objective: {:.5f}".format(it, f_new))
            else:
                if adaptive_restart == "gradient":
                    inner = _inner_real(y - x, x - x_old)
                    if inner > 0:
                        t = 1.0
                        z = np.array(x, copy=True)
                elif adaptive_restart == "function" and not monotonic and f_new > f_prev:
                    t = 1.0
                    z = np.array(x, copy=True)

                t0 = t
                t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
                z = x + ((t0 - 1.0) / t) * (x - x_old)

            f_prev = f_new

            if verbose and it % 10 == 0 and progress:
                print("Iteration: {}  objective: {:.5f}".format(it, f_new))

            if get_lambda and set_lambda and get_lambda() is not None:
                new_lambda = get_lambda() - noise
                if new_lambda > 0.0:
                    set_lambda(new_lambda)
                else:
                    if verbose:
                        print("Exit due to negative regularization parameter")
                    break

        return _f_value(F, x), x
