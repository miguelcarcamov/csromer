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
    phi: Union[np.ndarray, "da.Array"] = None
    data: Union[np.ndarray, "da.Array"] = None
    cellsize: float = None
    rmtf_fwhm: float = None
    max_recovered_width: float = None
    max_faraday_depth: float = None
    n: int = None

    def __init__(self, phi=None, cellsize=None, data=None):
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
    def data(self):
        return self.__data

    @data.setter
    def data(self, val):
        if val is not None:
            self.__data = val
            self.__n = length_of(val)
        else:
            self.__data = None

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, val):
        self.__n = val

    def calculate_cellsize(
        self,
        dataset: Dataset = None,
        oversampling=None,
        cellsize=None,
        set_size_pow_2=False,
        verbose=True,
    ):

        if dataset is not None:
            l2 = asnumpy(dataset.lambda2)
            w = asnumpy(dataset.w) if dataset.w is not None else np.ones(len(l2))
            l2_nonzero = l2[np.nonzero(w)]
            if len(l2_nonzero) == 0:
                l2_nonzero = l2
            l2_min = float(np.min(l2_nonzero))
            l2_max = float(np.max(dataset.lambda2) if not is_dask_array(dataset.lambda2) else np.max(l2))

            delta_phi_fwhm = 2.0 * np.sqrt(3.0) / (l2_max - l2_min)  # FWHM of the FPSF
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
            self.phi = self.cellsize * np.arange(-(self.n / 2), (self.n / 2), 1)
            self.data = np.zeros_like(self.phi, dtype=np.complex64)

    def calculate_sparsity(self):
        data_np = asnumpy(self.data)
        if data_np.dtype == np.complex64 or data_np.dtype == np.complex128:
            n = 2 * len(data_np)
            non_zeros = len(np.nonzero(data_np.real)[0]) + len(np.nonzero(data_np.imag)[0])
        else:
            n = len(data_np)
            non_zeros = len(np.nonzero(data_np)[0])
        return 100.0 * (1.0 - (non_zeros / n))

    def complex_data_to_real(self):
        """Convert Faraday depth from complex (n_phi,) to real stacked [real, imag] (2n). For external use (e.g. real-only optimizers)."""
        d = self.data
        dt = d.dtype if hasattr(d, "dtype") else getattr(asnumpy(d), "dtype", None)
        if dt == np.complex64 or dt == np.complex128:
            self.data = complex_to_real(d)
        else:
            raise TypeError("Parameter data is not complex64")

    def real_data_to_complex(self):
        """Convert Faraday depth from real stacked [real, imag] (2n) to complex (n_phi,). For external use after real-only optimization."""
        d = self.data
        dt = d.dtype if hasattr(d, "dtype") else getattr(asnumpy(d), "dtype", None)
        if dt == np.float32 or dt == np.float64:
            self.data = real_to_complex(d)
        else:
            raise ValueError("Parameter data is not real")

    def convolve(self, x=None, rmtf_fwhm=None):
        if rmtf_fwhm is None:
            rmtf_fwhm = self.rmtf_fwhm

        rmtf_fwhm_pixels = int(np.round(rmtf_fwhm / self.cellsize))
        val_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sigma_x = rmtf_fwhm / val_fwhm
        sigma_x_pixels = int(np.round(sigma_x / self.cellsize))

        print(
            "Convolving with Gaussian kernel where FWHM {0:2.3f} rad/m^2 - pixels {1}, sigma {2:2.3f} rad/m^2 - pixels {3}"
            .format(rmtf_fwhm, rmtf_fwhm_pixels, sigma_x, sigma_x_pixels)
        )

        clean_beam = Gaussian1DKernel(stddev=sigma_x_pixels)
        clean_beam_array = clean_beam.array

        data_src = self.data if x is None else x
        data_np = asnumpy(data_src)

        q_stokes = sci_signal.convolve(
            data_np.real, clean_beam_array, mode="same", method="fft"
        )
        u_stokes = sci_signal.convolve(
            data_np.imag, clean_beam_array, mode="same", method="fft"
        )
        p_stokes = q_stokes + 1j * u_stokes
        return p_stokes
