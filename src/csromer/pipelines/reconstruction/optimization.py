"""
CS-ROMER reconstructor: pipeline-based reconstruction by optimization.

Pipeline steps: L2Zero → BuildParameter → BuildMeasurementOperator →
DefaultObjective → DefaultOptimizer → Flag → DirtyMap → DirtyStats →
Optimization → Restoration → RestoredStats.

Pass pre-built instances (parameter, measurement_operator, objective_factory,
optimizer_factory) or use defaults via oversampling / measurement_operator_kind /
lambda_l_norm / wavelet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from csromer.dictionaries import Wavelet
from csromer.reconstruction import Parameter
from csromer.transformers.flaggers.flagger import Flagger

from .base import PipelineFaradayReconstructor
from .optimizer_factories import make_cg_optimizer, make_fista_optimizer
from .steps import (
    BuildMeasurementOperatorStep,
    BuildParameterStep,
    Clean1DStep,
    DefaultObjectiveFactoryStep,
    DefaultOptimizerFactoryStep,
    DirtyMapStep,
    DirtyStatsStep,
    FlagDataStep,
    L2ZeroStep,
    OptimizationStep,
    RestorationStep,
    RestoredStatsStep,
)

if TYPE_CHECKING:
    from csromer.transformers.measurement_operator import MeasurementOperator


@dataclass(init=True, repr=True)
class CSROMERReconstructorWrapper(PipelineFaradayReconstructor):
    """
    Pipeline-based reconstructor: inject parameter, measurement operator, objective, optimizer.

    Pipeline steps run in order; steps that see None will build defaults (parameter,
    measurement_operator, objective_factory, optimizer_factory).

    Attributes (injection; pass instances or leave None for defaults):
        dataset: Polarization dataset (required).
        parameter: Faraday depth parameter. If None, built in pipeline from dataset + oversampling.
        measurement_operator: Forward/adjoint operator. If None, built from measurement_operator_kind.
        objective_factory: (measurement_operator, parameter) -> OFunction. If None, ChiSquared + L1.
        optimizer_factory: (parameter, F_obj) -> optimizer with .run(). If None, FISTA.
        flagger: Optional flagger for data quality.

    Optional defaults (used when corresponding injection is None):
        oversampling (7.0), measurement_operator_kind ("direct"), lambda_l_norm (0.0),
        wavelet (None), calculate_l2_zero (False). When kind is "gridded": gridding_kernel
        ("kaiser" or "box"), gridding_kernel_half_width (4.0), gridding_kernel_beta (2.5).
        When adaptive_lambda is True: target_chi2 (1.0), chi2_target_rel_tol (0.1),
        lambda_update_gamma (0.5), max_lambda_updates (5).

    After reconstruct(): fd_dirty, fd_model, fd_residual, fd_restored, rm_dirty,
        rm_model, rm_restored, second_moment, and error/quadratic-interp attributes.
    """
    parameter: Parameter = None
    measurement_operator: "MeasurementOperator" = None
    objective_factory: Callable = None
    optimizer_factory: Callable = None
    flagger: Flagger = None
    oversampling: float = 7.0
    measurement_operator_kind: str = "direct"
    lambda_l_norm: float = 0.0
    wavelet: Wavelet = None
    calculate_l2_zero: bool = False
    # Gridding options (when measurement_operator_kind == "gridded")
    gridding_kernel: str = "kaiser"  # "kaiser" or "box"
    gridding_kernel_half_width: float = 4.0
    gridding_kernel_beta: float = 2.5
    # Optional adaptive-λ configuration (for experimental outer-loop lambda selection)
    adaptive_lambda: bool = False
    target_chi2: float = 1.0
    chi2_target_rel_tol: float = 0.05  # accept when chi2 <= target_chi2 * (1 + this)
    lambda_update_gamma: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = np.inf
    max_lambda_updates: int = 5

    def get_steps(self):
        return [
            L2ZeroStep(),
            BuildParameterStep(),
            BuildMeasurementOperatorStep(),
            DefaultObjectiveFactoryStep(),
            DefaultOptimizerFactoryStep(),
            FlagDataStep(),
            DirtyMapStep(),
            DirtyStatsStep(),
            OptimizationStep(),
            RestorationStep(),
            RestoredStatsStep(),
        ]


@dataclass(init=True, repr=True)
class CLEANReconstructorWrapper(PipelineFaradayReconstructor):
    """
    Pipeline-based reconstructor using 1D CLEAN from the dirty map (no optimization).

    Steps: L2Zero → BuildParameter → BuildMeasurementOperator → Flag →
    DirtyMap → DirtyStats → Clean1DStep → Restoration → RestoredStats.

    Same result attributes as CSROMERReconstructorWrapper (fd_dirty, fd_model,
    fd_restored, fd_residual, rm_*, second_moment, etc.).
    """
    parameter: Parameter = None
    measurement_operator: "MeasurementOperator" = None
    flagger: Flagger = None
    oversampling: float = 7.0
    measurement_operator_kind: str = "direct"
    gridding_kernel: str = "kaiser"
    gridding_kernel_half_width: float = 4.0
    gridding_kernel_beta: float = 2.5
    clean_gain: float = 0.2
    clean_maxiter: int = 500
    clean_threshold: float | None = None
    clean_n_sigma: float | None = None

    def get_steps(self):
        return [
            L2ZeroStep(),
            BuildParameterStep(),
            BuildMeasurementOperatorStep(),
            FlagDataStep(),
            DirtyMapStep(),
            DirtyStatsStep(),
            Clean1DStep(
                gain=self.clean_gain,
                maxiter=self.clean_maxiter,
                threshold=self.clean_threshold,
                n_sigma=self.clean_n_sigma,
            ),
            RestorationStep(),
            RestoredStatsStep(),
        ]
