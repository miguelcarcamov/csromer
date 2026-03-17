"""
Reconstruction (FISTA, CG, or CLEAN) for a single dataset.
"""

from __future__ import annotations

import numpy as np

from csromer.utils.array_utils import asnumpy
from csromer.pipelines import (
    CLEANReconstructorWrapper,
    CSROMERReconstructorWrapper,
    make_cg_optimizer,
    make_fista_optimizer,
)


def _estimate_lambda_l1(
    dataset,
    k: float = 0.3,
) -> float:
    """
    Heuristic L1 regularization strength based on noise level (data space).

    Uses:
        lambda = k * sigma_eff * sqrt(2*log(m)),
    where sigma_eff is a robust per-channel noise estimate and m is the number
    of channels. Log scaling keeps lambda moderate. Call with the dataset used
    for the fit (e.g. after gridding) so m and sigma match.
    """
    sigma = getattr(dataset, "sigma", None)
    sigma_eff = None
    if sigma is not None:
        sigma_np = np.asarray(asnumpy(sigma))
        positive = sigma_np[sigma_np > 0]
        if positive.size > 0:
            sigma_eff = float(np.median(positive))
    if sigma_eff is None:
        sigma_eff = float(getattr(dataset, "theo_noise", 0.0) or 0.0)
    if sigma_eff <= 0.0:
        return 0.5

    m = int(getattr(dataset, "m", 0) or 0)
    if m <= 0 and getattr(dataset, "data", None) is not None:
        m = len(asnumpy(dataset.data))
    if m <= 0:
        return 0.5

    lam = k * sigma_eff * np.sqrt(2.0 * np.log(max(m, 2)))
    return float(lam)


def run_csromer_reconstruction(
    source,
    oversampling: float = 4.0,
    maxiter: int = 100,
    reconstructor: str = "cg",
    k_lambda: float = 0.3,
    target_residual_sigma: float | None = None,
    compute_sigma_fd: bool = False,
    fd_accept_n_sigma: float | None = None,
):
    """
    Run reconstruction on a csromer Dataset.

    Args:
        source: Dataset (e.g. simulated source).
        oversampling: Oversampling factor for Faraday depth grid.
        maxiter: Maximum iterations (FISTA or CG); CLEAN uses its own maxiter.
        reconstructor: "csromer" (FISTA + L1), "cg", or "clean".
        k_lambda: Multiplier for the noise-based L1 strength when using FISTA.
        target_residual_sigma: Target residual level in sigma (e.g. 1 for noise, 5 for ~5σ).
            Sets target_chi2 = 0.5*value² for FISTA adaptive λ. Default 1.0. Ignored for cg/clean.
        compute_sigma_fd: If True, compute FD-space noise σ_fd from data Σ_d via A^H Σ_d A
            (Hutchinson). Used for CLEAN threshold and FD-panel noise lines; data sigma unchanged.
        fd_accept_n_sigma: Require max|fd_residual| <= this value * σ_fd to accept λ in FISTA
            (e.g. 5). Implies compute_sigma_fd. None = chi2-only acceptance.

    Returns:
        Reconstructor wrapper instance after reconstruct().
    """
    rec = reconstructor.lower()
    if fd_accept_n_sigma is not None and not compute_sigma_fd:
        compute_sigma_fd = True
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
            compute_sigma_fd=compute_sigma_fd,
        )
    elif rec == "cg":
        optimizer_factory = make_cg_optimizer(maxiter=maxiter, tol=1e-12, verbose=True)
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=1e-3,
            optimizer_factory=optimizer_factory,
            compute_sigma_fd=compute_sigma_fd,
        )
    else:
        # χ² = (1/2)*sum(w|r|²)/n_eff ⇒ E[χ²] = (1/2)*(r/σ)² per DoF. So 5σ → target 12.5.
        _target_sigma = target_residual_sigma if target_residual_sigma is not None else 1.0
        _target_chi2 = 0.5 * float(_target_sigma ** 2)

        # When σ_fd is computed (Hutchinson), use FD acceptance by default with same sigma as data target.
        _fd_accept_n_sigma = fd_accept_n_sigma
        if compute_sigma_fd and _fd_accept_n_sigma is None:
            _fd_accept_n_sigma = _target_sigma

        def _lambda_estimator(dataset):
            return _estimate_lambda_l1(dataset, k=k_lambda)

        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_estimator=_lambda_estimator,
            adaptive_lambda=True,
            target_chi2=_target_chi2,
            lambda_update_gamma=0.5,
            max_lambda_updates=50,
            optimizer_factory=make_fista_optimizer(
                maxiter=maxiter,
                tol=1e-12,
                verbose=True,
                monotonic=True,
            ),
            compute_sigma_fd=compute_sigma_fd,
            fd_accept_n_sigma=_fd_accept_n_sigma,
        )
    recon.reconstruct()
    return recon
