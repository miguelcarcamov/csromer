"""
Reconstruction (FISTA, CG, or CLEAN) for a single dataset.
"""

from __future__ import annotations

import numpy as np

from csromer.utils.array_utils import asnumpy
from csromer.pipelines import (
    CLEANReconstructorWrapper,
    CSROMERReconstructorWrapper,
    build_parameter,
    make_cg_optimizer,
    make_fista_optimizer,
)


def _estimate_lambda_l1(
    source,
    oversampling: float,
    k: float = 0.3,
) -> float:
    """
    Heuristic L1 regularization strength based on noise level.

    Uses a Donoho–Johnstone-style universal threshold:
        lambda ≈ k * sigma_eff * sqrt(2 log n),
    where sigma_eff is a robust per-channel noise estimate and n is the number
    of Faraday-depth coefficients (matching the grid that will be used).
    """
    sigma = getattr(source, "sigma", None)
    sigma_eff = None
    if sigma is not None:
        sigma_np = np.asarray(asnumpy(sigma))
        positive = sigma_np[sigma_np > 0]
        if positive.size > 0:
            sigma_eff = float(np.median(positive))
    if sigma_eff is None:
        sigma_eff = float(getattr(source, "theo_noise", 0.0) or 0.0)
    if sigma_eff <= 0.0:
        return 0.5

    param_tmp = build_parameter(dataset=source, oversampling=oversampling)
    n_coeff = int(getattr(param_tmp, "n", 0) or 0)
    if n_coeff <= 0:
        return 0.5

    lam = k * sigma_eff * np.sqrt(2.0 * np.log(n_coeff))
    return float(lam)


def run_csromer_reconstruction(
    source,
    oversampling: float = 4.0,
    maxiter: int = 200,
    reconstructor: str = "cg",
    k_lambda: float = 0.3,
):
    """
    Run reconstruction on a csromer Dataset.

    Args:
        source: Dataset (e.g. simulated source).
        oversampling: Oversampling factor for Faraday depth grid.
        maxiter: Maximum iterations (FISTA or CG); CLEAN uses its own maxiter.
        reconstructor: "csromer" (FISTA + L1), "cg", or "clean".
        k_lambda: Multiplier for the noise-based L1 strength when using FISTA.

    Returns:
        Reconstructor wrapper instance after reconstruct().
    """
    rec = reconstructor.lower()
    if rec == "clean":
        source.l2_ref = source.calculate_l2ref()
        recon = CLEANReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            clean_gain=0.1,
            clean_maxiter=maxiter,
            clean_threshold=None,
            clean_n_sigma=3.0,
        )
    elif rec == "cg":
        optimizer_factory = make_cg_optimizer(maxiter=maxiter, tol=1e-12, verbose=True)
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=1e-3,
            optimizer_factory=optimizer_factory,
        )
    else:
        lambda_init = _estimate_lambda_l1(source=source, oversampling=oversampling, k=k_lambda)
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=lambda_init,
            adaptive_lambda=True,
            target_chi2=1.0,
            lambda_update_gamma=0.5,
            max_lambda_updates=50,
            optimizer_factory=make_fista_optimizer(
                maxiter=maxiter,
                tol=1e-12,
                verbose=True,
                monotonic=True,
            ),
        )
    recon.reconstruct()
    return recon
