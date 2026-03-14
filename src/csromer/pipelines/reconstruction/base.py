"""
Base classes for Faraday depth reconstructors.

Pipeline-based: subclasses implement get_steps(). reconstruct() runs the pipeline.
PipelineFaradayReconstructor holds shared parameter/measurement_operator/flagger
and result attributes (fd_dirty, fd_restored, rm_*, etc.) plus shared methods
(config_fd_space, flag_dataset, get_rm, ...) so CS-ROMER and CLEAN avoid duplication.
"""
from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from csromer.base import Dataset
from csromer.pipelines.core import run_pipeline

if TYPE_CHECKING:
    from csromer.reconstruction import Parameter
    from csromer.transformers.flaggers.flagger import Flagger
    from csromer.transformers.measurement_operator import MeasurementOperator


@dataclass(init=True, repr=True)
class FaradayReconstructorWrapper(metaclass=ABCMeta):
    """
    Base class for Faraday depth reconstructors (pipeline pattern).

    Subclasses that use the pipeline implement get_steps();
    reconstruct() runs run_pipeline(self, self.get_steps()).
    """
    dataset: Dataset = None

    def get_steps(self):
        """Return ordered list of pipeline steps. Override in subclasses."""
        return []

    def reconstruct(self):
        """Run reconstruction. Default: run pipeline from get_steps()."""
        steps = self.get_steps()
        if steps:
            run_pipeline(self, steps)

    def config_fd_space(self, cellsize=None, oversampling=None):
        """Configure Faraday depth space. Override in subclasses that support it."""
        pass

    def calculate_second_moment(self) -> float:
        """Second moment of model (width measure). Override in subclasses."""
        return 0.0


@dataclass(init=True, repr=True)
class PipelineFaradayReconstructor(FaradayReconstructorWrapper):
    """
    Base for pipeline reconstructors that use parameter, measurement_operator, flagger.

    Holds shared result attributes (fd_dirty, fd_restored, rm_*, etc.) and shared
    methods (config_fd_space, flag_dataset, get_dirty_faraday_depth, get_rmtf,
    get_rm, calculate_second_moment). Subclasses (CS-ROMER, CLEAN) add their
    specific options and get_steps().
    """
    parameter: "Parameter | None" = None
    measurement_operator: "MeasurementOperator | None" = None
    flagger: "Flagger | None" = None

    # Result attributes (set by pipeline steps)
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

    def config_fd_space(self, cellsize: float | None = None, oversampling: float | None = None):
        if cellsize is not None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        elif oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, oversampling=oversampling)
        elif cellsize is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        else:
            raise ValueError("Provide cellsize or oversampling")

    def flag_dataset(self, flagger=None):
        f = flagger if flagger is not None else self.flagger
        return f.run()

    def get_dirty_faraday_depth(self) -> np.ndarray:
        return self.measurement_operator.dirty_spectrum(self.dataset.data)

    def get_rmtf(self) -> np.ndarray:
        return self.measurement_operator.RMTF()

    def get_rm(self, fd_data: np.ndarray) -> float:
        from csromer.utils.array_utils import asnumpy
        fd_abs = np.asarray(asnumpy(np.abs(fd_data)))
        idx = int(np.argmax(fd_abs))
        phi = np.asarray(asnumpy(self.parameter.phi))
        return float(phi[idx])

    def calculate_second_moment(self) -> float:
        if self.fd_model is None:
            return 0.0
        from .reconstruction_stats import calculate_second_moment as _second_moment
        return _second_moment(self.parameter.phi, self.fd_model)
