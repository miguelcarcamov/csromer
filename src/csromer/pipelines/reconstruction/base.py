"""
Base class for Faraday depth reconstructors.

Pipeline-based: subclasses implement get_steps() and optionally
config_fd_space / calculate_second_moment. reconstruct() runs the pipeline.
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from csromer.base import Dataset
from csromer.pipelines.core import run_pipeline


@dataclass(init=True, repr=True)
class FaradayReconstructorWrapper(metaclass=ABCMeta):
    """
    Base class for Faraday depth reconstructors (pipeline pattern).

    Subclasses that use the pipeline implement get_steps() and
    reconstruct() runs run_pipeline(self, self.get_steps()).
    Subclasses that do not use the pipeline override reconstruct() directly.

    Attributes:
        dataset: Dataset object with polarization data
    """
    dataset: Dataset = None

    def get_steps(self):
        """
        Return ordered list of pipeline steps. Override in subclasses.
        Return [] if this reconstructor does not use the pipeline.
        """
        return []

    def reconstruct(self):
        """
        Run reconstruction. Default: run pipeline from get_steps().
        Override to use a different strategy (e.g. pol-angle gradient).
        """
        steps = self.get_steps()
        if steps:
            run_pipeline(self, steps)
        # else: subclass overrides reconstruct() and doesn't call super()

    def config_fd_space(self, cellsize=None, oversampling=None):
        """
        Configure Faraday depth space (grid and cellsize).
        Override in subclasses that support it.
        """
        pass

    def calculate_second_moment(self) -> float:
        """
        Calculate second moment of model (width measure).
        Override in subclasses. Returns 0.0 by default.
        """
        return 0.0
