"""
CS-ROMER reconstruction (FISTA or CG) for a single dataset.
"""

from __future__ import annotations

from csromer.pipelines import (
    CSROMERReconstructorWrapper,
    make_cg_optimizer,
    make_fista_optimizer,
)


def run_csromer_reconstruction(
    source,
    oversampling: float = 4.0,
    maxiter: int = 100,
    reconstructor: str = "cg",
):
    """
    Run reconstruction on a csromer Dataset.

    Args:
        source: Dataset (e.g. simulated source).
        oversampling: Oversampling factor for Faraday depth grid.
        maxiter: Maximum iterations (FISTA or CG).
        reconstructor: "csromer" (FISTA + L1) or "cg".

    Returns:
        CSROMERReconstructorWrapper instance after reconstruct().
    """
    if reconstructor.lower() == "cg":
        optimizer_factory = make_cg_optimizer(maxiter=maxiter, tol=1e-12, verbose=True)
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=1e-3,
            optimizer_factory=optimizer_factory,
        )
    else:
        recon = CSROMERReconstructorWrapper(
            dataset=source,
            oversampling=oversampling,
            measurement_operator_kind="gridded",
            lambda_l_norm=0.5,
            adaptive_lambda=True,
            target_chi2=1.0,
            lambda_update_gamma=0.5,
            max_lambda_updates=20,
            optimizer_factory=make_fista_optimizer(
                maxiter=maxiter,
                tol=1e-12,
                verbose=True,
                monotonic=True,
            ),
        )
    recon.reconstruct()
    return recon
