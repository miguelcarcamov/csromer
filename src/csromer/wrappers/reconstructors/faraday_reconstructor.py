"""
Base class for Faraday depth reconstructors.

Abstract interface that all reconstructors must implement.
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...base import Dataset


@dataclass(init=True, repr=True)
class FaradayReconstructorWrapper(metaclass=ABCMeta):
    """
    Base class for Faraday depth reconstructors.
    
    Abstract interface that defines the contract for reconstruction classes.
    Subclasses must implement config_fd_space, reconstruct, and calculate_second_moment.
    
    Attributes:
        dataset: Dataset object with polarization data
    """
    dataset: Dataset = None

    @abstractmethod
    def config_fd_space(self):
        """
        Configure Faraday depth space (grid and cellsize).
        
        Abstract method: subclasses must implement. Sets up phi grid and cellsize
        based on dataset properties.
        """
        pass

    @abstractmethod
    def reconstruct(self):
        """
        Run reconstruction.
        
        Abstract method: subclasses must implement. Performs the full reconstruction
        pipeline and sets result attributes.
        """
        pass

    @abstractmethod
    def calculate_second_moment(self) -> float:
        """
        Calculate second moment of model (width measure).
        
        Abstract method: subclasses must implement. Computes weighted second moment
        around first moment.
        
        Returns:
            Second moment (rad²/m⁴)
        """
        pass
