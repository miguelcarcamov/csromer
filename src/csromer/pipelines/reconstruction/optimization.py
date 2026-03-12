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
from csromer.utils.array_utils import asnumpy

from .base import FaradayReconstructorWrapper
from .optimizer_factories import make_cg_optimizer, make_fista_optimizer
from .reconstruction_stats import calculate_second_moment
from .steps import (
    BuildMeasurementOperatorStep,
    BuildParameterStep,
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
class CSROMERReconstructorWrapper(FaradayReconstructorWrapper):
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

    coefficients: np.ndarray = field(init=False, default=None)
    fd_dirty: np.ndarray = field(init=False, default=None)
    rm_dirty: float = field(init=False, default=None)
    rm_dirty_error: float = field(init=False, default=None)
    rm_dirty_quadratic_interpolation: float = field(init=False, default=None)
    dirty_peak_quadratic_interpolation: float = field(init=False, default=None)
    rm_dirty_quadratic_interpolation_error: float = field(init=False, default=None)
    fd_model: np.ndarray = field(init=False, default=None)
    rm_model: float = field(init=False, default=None)
    fd_residual: np.ndarray = field(init=False, default=None)
    fd_restored: np.ndarray = field(init=False, default=None)
    rm_restored: float = field(init=False, default=None)
    rm_restored_error: float = field(init=False, default=None)
    rm_restored_quadratic_interpolation: float = field(init=False, default=None)
    restored_peak_quadratic_interpolation: float = field(init=False, default=None)
    rm_restored_quadratic_interpolation_error: float = field(init=False, default=None)
    second_moment: float = field(init=False, default=None)

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

    def config_fd_space(self, cellsize: float = None, oversampling: float = None):
        if cellsize is not None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        elif oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, oversampling=oversampling)
        elif cellsize is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        else:
            raise ValueError("Provide cellsize or oversampling")

    def flag_dataset(self, flagger: Flagger = None) -> tuple:
        f = flagger if flagger is not None else self.flagger
        return f.run()

    def get_dirty_faraday_depth(self) -> np.ndarray:
        return self.measurement_operator.dirty_spectrum(self.dataset.data)

    def get_rmtf(self) -> np.ndarray:
        return self.measurement_operator.RMTF()

    def get_rm(self, fd_data: np.ndarray) -> float:
        fd_abs = np.asarray(asnumpy(np.abs(fd_data)))
        idx = int(np.argmax(fd_abs))
        phi = np.asarray(asnumpy(self.parameter.phi))
        return float(phi[idx])

    def calculate_second_moment(self) -> float:
        if self.fd_model is None:
            return 0.0
        return calculate_second_moment(self.parameter.phi, self.fd_model)
