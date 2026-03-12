"""
Parameter: Faraday depth space configuration and data.

Manages phi grid, cellsize, RMTF properties, and data conversion between
complex and real representations for optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy.convolution import Gaussian1DKernel
from scipy import signal as sci_signal

from ..utils import complex_to_real, next_power_2, real_to_complex
from ..utils.array_utils import asnumpy, is_dask_array, length_of

if TYPE_CHECKING:
    from ..base import Dataset

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(init=False, repr=True)
class Parameter:
    """
    Faraday depth space parameter configuration.
    
    Manages the phi grid (Faraday depth axis), cellsize, RMTF properties,
    and data storage. Supports conversion between complex and real representations
    for optimizers that require real-only arrays.
    
    Attributes:
        phi: Faraday depth grid (rad/m²)
        data: Complex Faraday depth spectrum (or real stacked [real, imag])
        cellsize: Grid spacing (rad/m²)
        rmtf_fwhm: RMTF FWHM (rad/m²)
        max_recovered_width: Maximum recoverable structure width (rad/m²)
        max_faraday_depth: Maximum Faraday depth with >50% sensitivity (rad/m²)
        n: Number of grid points
    """
    phi: Union[np.ndarray, "da.Array"] = None
    data: Union[np.ndarray, "da.Array"] = None
    cellsize: float = None
    rmtf_fwhm: float = None
    max_recovered_width: float = None
    max_faraday_depth: float = None
    n: int = None

    def __init__(self, phi=None, cellsize=None, data=None):
        """
        Initialize Parameter.
        
        Args:
            phi: Faraday depth grid (rad/m²). If None, will be computed by calculate_cellsize.
            cellsize: Grid spacing (rad/m²). If None, will be computed by calculate_cellsize.
            data: Initial data array (complex or real stacked)
        """
        self.phi = phi
        self.data = data
        self.cellsize = cellsize

        self.rmtf_fwhm = 0.0
        self.max_recovered_width = 0.0
        self.max_faraday_depth = 0.0

        if self.phi is not None:
            self.n = length_of(phi)
        elif self.data is not None:
            self.n = length_of(data)
        else:
            self.n = 0

    @property
    def data(self) -> Union[np.ndarray, "da.Array", None]:
        """Faraday depth spectrum data (complex or real stacked)."""
        return self.__data

    @data.setter
    def data(self, val):
        """
        Set data array and update n.
        
        Args:
            val: Data array (complex or real stacked)
        """
        if val is not None:
            self.__data = val
            self.__n = length_of(val)
        else:
            self.__data = None

    @property
    def n(self) -> int:
        """Number of grid points."""
        return self.__n

    @n.setter
    def n(self, val):
        """Set number of grid points."""
        self.__n = val

    def calculate_cellsize(
        self,
        dataset: "Dataset" = None,
        oversampling=None,
        cellsize=None,
        set_size_pow_2=False,
        verbose=True,
    ):
        """
        Calculate optimal cellsize and phi grid from dataset.
        
        Computes RMTF properties (FWHM, max recovered width, max Faraday depth)
        and sets phi grid with appropriate cellsize and size.
        
        Args:
            dataset: Dataset with lambda² coverage
            oversampling: Oversampling factor (default: 8.0)
            cellsize: Fixed cellsize to use (overrides computed value)
            set_size_pow_2: If True, round n to next power of 2
            verbose: Print RMTF properties (default: True)
        """
        if dataset is not None:
            l2 = asnumpy(dataset.lambda2)
            w = asnumpy(dataset.w) if dataset.w is not None else np.ones(len(l2))
            l2_nonzero = l2[np.nonzero(w)]
            if len(l2_nonzero) == 0:
                l2_nonzero = l2
            l2_min = float(np.min(l2_nonzero))
            l2_max = float(np.max(dataset.lambda2) if not is_dask_array(dataset.lambda2) else np.max(l2))

            # Single source of truth: RMTF FWHM from Dataset (full vs nominal by l2_ref)
            delta_phi_fwhm = dataset.delta_phi
            if delta_phi_fwhm is None:
                # Fallback when Dataset has no lambda2: nominal if spread > 0, else full (avoids div by zero)
                if l2_max > l2_min:
                    delta_phi_fwhm = 2.0 * np.sqrt(3.0) / (l2_max - l2_min)
                else:
                    delta_phi_fwhm = 2.0 / (l2_max + l2_min)
            delta_phi_theo = np.pi / l2_min

            delta_phi = min(delta_phi_fwhm, delta_phi_theo)
            phi_max = np.sqrt(3) / dataset.delta_l2_mean
            phi_max = max(phi_max, delta_phi_fwhm * 10.0)

            self.rmtf_fwhm = delta_phi_fwhm
            self.max_recovered_width = delta_phi_theo
            self.max_faraday_depth = phi_max

            if verbose:
                print("FWHM of the main peak of the RMTF: {0:.3f} rad/m^2".format(self.rmtf_fwhm))
                print(
                    "Maximum recovered width structure: {0:.3f} rad/m^2".format(
                        self.max_recovered_width
                    )
                )
                print(
                    "Maximum Faraday Depth to which one has more than 50% sensitivity: {0:.3f}".
                    format(self.max_faraday_depth)
                )

            if oversampling is None:
                oversampling = 8.

            if cellsize is None:
                phi_r = delta_phi / oversampling
            else:
                phi_r = cellsize

            temp = np.int32(np.floor(2 * phi_max / phi_r))

            if set_size_pow_2:
                self.n = next_power_2(temp)
            else:
                self.n = int(temp - np.mod(temp, 32))

            self.cellsize = 2 * phi_max / self.n
            self.phi = self.cellsize * (np.arange(self.n) - self.n // 2)
            self.data = np.zeros_like(self.phi, dtype=np.complex64)

    def calculate_sparsity(self) -> float:
        """
        Calculate sparsity percentage of data.
        
        Returns:
            Sparsity percentage (0-100): 100 * (1 - nonzeros / total_elements)
        """
        data_np = asnumpy(self.data)
        if data_np.dtype == np.complex64 or data_np.dtype == np.complex128:
            n = 2 * len(data_np)
            non_zeros = len(np.nonzero(data_np.real)[0]) + len(np.nonzero(data_np.imag)[0])
        else:
            n = len(data_np)
            non_zeros = len(np.nonzero(data_np)[0])
        return 100.0 * (1.0 - (non_zeros / n))

    def complex_data_to_real(self):
        """
        Convert Faraday depth from complex (n_phi,) to real stacked [real, imag] (2n).
        
        For use with real-only optimizers. Converts complex array to [real, imag] stacked.
        
        Raises:
            TypeError: If data is not complex
        """
        d = self.data
        dt = d.dtype if hasattr(d, "dtype") else getattr(asnumpy(d), "dtype", None)
        if dt == np.complex64 or dt == np.complex128:
            self.data = complex_to_real(d)
        else:
            raise TypeError("Parameter data is not complex64")

    def real_data_to_complex(self):
        """
        Convert Faraday depth from real stacked [real, imag] (2n) to complex (n_phi,).
        
        For use after real-only optimization. Converts [real, imag] stacked array back to complex.
        
        Raises:
            ValueError: If data is not real
        """
        d = self.data
        dt = d.dtype if hasattr(d, "dtype") else getattr(asnumpy(d), "dtype", None)
        if dt == np.float32 or dt == np.float64:
            self.data = real_to_complex(d)
        else:
            raise ValueError("Parameter data is not real")

    def convolve(self, x=None, rmtf_fwhm=None) -> np.ndarray:
        """
        Convolve Faraday depth spectrum with Gaussian restore beam (max=1).

        Convolves real and imaginary parts separately; multiple peaks stay
        separated. Output is Jy/phi_pixel; caller multiplies by
        (rmtf_fwhm / cellsize) to get Jy/rmtf for restoration.

        Args:
            x: Input array (default: self.data), Jy/phi_pixel
            rmtf_fwhm: RMTF FWHM for kernel (default: self.rmtf_fwhm)

        Returns:
            Convolved spectrum (complex), Jy/phi_pixel
        """
        if rmtf_fwhm is None:
            rmtf_fwhm = self.rmtf_fwhm

        val_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sigma_x = rmtf_fwhm / val_fwhm
        # Use float sigma so kernel FWHM in physical space equals rmtf_fwhm (integer rounding
        # would oversmooth and merge closely spaced components)
        sigma_x_pixels = max(1.0, sigma_x / self.cellsize)

        print(
            "Convolving with Gaussian kernel where FWHM {0:2.3f} rad/m^2 - sigma {1:2.3f} rad/m^2 - sigma_pixels {2:.4f}"
            .format(rmtf_fwhm, sigma_x, sigma_x_pixels)
        )

        clean_beam = Gaussian1DKernel(stddev=sigma_x_pixels)
        clean_beam_array = np.asarray(clean_beam.array, dtype=np.float64).ravel()
        # Normalize so max=1: conserves peak (flux) so conv_model peak ≈ model peak
        k_max = float(np.max(clean_beam_array))
        if k_max > 0:
            clean_beam_array = clean_beam_array / k_max
        print("  [convolve] kernel len=%d sum=%.6f max=%.6f (max=1, peak-conserving)" % (
            len(clean_beam_array), float(np.sum(clean_beam_array)), float(np.max(clean_beam_array))))

        data_src = x if x is not None else self.data
        data_np = np.asarray(asnumpy(data_src), dtype=np.complex128)
        print("  [convolve] input x is None=%s  peak|data_np|=%.6e  sum|data_np|=%.6e" % (
            x is None, float(np.max(np.abs(data_np))), float(np.sum(np.abs(data_np)))))

        # Convolve real and imaginary parts separately (keeps multiple peaks separated)
        q_stokes = sci_signal.convolve(data_np.real, clean_beam_array, mode="same", method="fft")
        u_stokes = sci_signal.convolve(data_np.imag, clean_beam_array, mode="same", method="fft")
        p_stokes = (q_stokes + 1j * u_stokes).astype(data_np.dtype)
        return p_stokes
