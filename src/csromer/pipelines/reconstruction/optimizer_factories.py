"""
Optimizer factories for the reconstructor.

Each function returns a callable (parameter, F_obj) -> optimizer with .run().
Use these when you do not pass your own optimizer_factory. Extend with
make_lbfgs_optimizer, etc., as needed.
"""
from __future__ import annotations

from typing import Callable

from csromer.optimization import FISTA, PolakRibiere


def make_fista_optimizer(
    maxiter: int = 500,
    tol: float = None,
    verbose: bool = True,
    step: float = None,
    monotonic: bool = False,
) -> Callable:
    """
    Return an optimizer factory for FISTA.
    The returned callable takes (parameter, F_obj) and returns a FISTA instance.
    """
    def factory(parameter, F_obj):
        kw = dict(
            guess_param=parameter,
            F_obj=F_obj,
            maxiter=maxiter,
            verbose=verbose,
            monotonic=monotonic,
        )
        if tol is not None:
            kw["tol"] = tol
        if step is not None:
            kw["step"] = step
        return FISTA(**kw)
    return factory


def make_cg_optimizer(
    method: type = PolakRibiere,
    maxiter: int = 500,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Callable:
    """
    Return an optimizer factory for Conjugate Gradient.
    The returned callable takes (parameter, F_obj) and returns a CG optimizer instance.
    """
    def factory(parameter, F_obj):
        grad_fun = None
        if F_obj.F and len(F_obj.F) == 1:
            grad_fun = F_obj.F[0].calculate_gradient
        return method(
            guess_param=parameter,
            F_obj=F_obj,
            grad_fun=grad_fun,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        )
    return factory
