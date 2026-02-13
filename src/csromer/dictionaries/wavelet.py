"""
Base wavelet transform class for sparse representation.

Abstract interface for wavelet transforms (discrete, undecimated, continuous).
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from pywt import Wavelet


@dataclass(init=True, repr=True)
class Wavelet(metaclass=ABCMeta):
    """
    Base class for wavelet transforms.
    
    Abstract interface for wavelet decomposition and reconstruction. Subclasses
    implement discrete, undecimated, or continuous wavelets.
    
    Attributes:
        wavelet_name: Name of wavelet (e.g. 'db4', 'haar')
        wavelet_level: Decomposition level
        mode: Boundary mode (e.g. 'symmetric', 'periodic')
        append_signal: Whether to append signal to coefficients
        ncoeffs: Number of coefficients (computed)
        n: Signal length (computed)
        wavelet: PyWavelets Wavelet object (computed)
        coeff_slices: Coefficient slice indices (computed)
    """
    wavelet_name: str = None
    wavelet_level: int = None
    mode: str = None
    append_signal: bool = None
    ncoeffs: int = field(init=False, default=0)
    n: int = field(init=False, default=0)
    wavelet: Wavelet = field(init=False, default=None)
    coeff_slices: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        """
        Post-initialization: validate wavelet_name.
        
        Raises:
            TypeError: If wavelet_name is not a string
        """
        if not isinstance(self.wavelet_name, str):
            raise TypeError("The wavelet name is not a string")

    @abstractmethod
    def calculate_max_level(self, x):
        """
        Calculate maximum decomposition level for signal x.
        
        Abstract method: subclasses must implement.
        
        Args:
            x: Input signal
            
        Returns:
            Maximum level (int)
        """
        return

    @abstractmethod
    def decompose(self, x):
        """
        Decompose signal into wavelet coefficients (real).
        
        Abstract method: subclasses must implement.
        
        Args:
            x: Input signal (real)
            
        Returns:
            Wavelet coefficients
        """
        return

    @abstractmethod
    def decompose_complex(self, x):
        """
        Decompose complex signal into wavelet coefficients.
        
        Abstract method: subclasses must implement. Decomposes real and imaginary
        parts separately.
        
        Args:
            x: Input signal (complex)
            
        Returns:
            Wavelet coefficients (complex)
        """
        return

    @abstractmethod
    def reconstruct(self, input_coeffs):
        """
        Reconstruct signal from wavelet coefficients (real).
        
        Abstract method: subclasses must implement.
        
        Args:
            input_coeffs: Wavelet coefficients
            
        Returns:
            Reconstructed signal (real)
        """
        return

    @abstractmethod
    def reconstruct_complex(self, input_coeffs):
        """
        Reconstruct complex signal from wavelet coefficients.
        
        Abstract method: subclasses must implement. Reconstructs real and imaginary
        parts separately.
        
        Args:
            input_coeffs: Wavelet coefficients (complex)
            
        Returns:
            Reconstructed signal (complex)
        """
        return
