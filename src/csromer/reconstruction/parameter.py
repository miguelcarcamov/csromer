from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.convolution import Gaussian1DKernel
from scipy import signal as sci_signal

from ..utils import complex_to_real, next_power_2, real_to_complex

if TYPE_CHECKING:
    from ..base import Dataset


@dataclass(init=False, repr=True)
class Parameter:
    phi: np.ndarray = None
    data: np.ndarray = None
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
            self.n = len(phi)
        elif self.data is not None:
            self.n = len(data)
        else:
            self.n = 0

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, val):
        if val is not None:
            self.__data = val
            self.__n = len(val)
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
            l2_min = np.min(dataset.lambda2[np.nonzero(dataset.lambda2)])
            l2_max = np.max(dataset.lambda2)

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
        if self.data.dtype == np.complex64 or self.data.dtype == np.complex128:
            n = 2 * len(self.data)
            non_zeros = len(np.nonzero(self.data.real)) + len(np.nonzero(self.data.imag))
        else:
            n = len(self.data)
            non_zeros = len(np.nonzero(self.data))
        return 100.0 * (1.0 - (non_zeros / n))

    def complex_data_to_real(self):
        if self.data.dtype == np.complex64 or self.data.dtype == np.complex128:
            self.data = complex_to_real(self.data)
        else:
            raise TypeError("Parameter data is not complex64")

    def real_data_to_complex(self):
        if self.data.dtype == np.float32 or self.data.dtype == np.float64:
            self.data = real_to_complex(self.data)
        else:
            raise ValueError("Parameter data is not real")

    def convolve(self, x=None, rmtf_fwhm=None):

        if rmtf_fwhm is None:
            rmtf_fwhm = self.rmtf_fwhm

        rmtf_fwhm_pixels = np.round(rmtf_fwhm / self.cellsize).astype(np.int32)
        val_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sigma_x = rmtf_fwhm / val_fwhm
        sigma_x_pixels = np.round(sigma_x / self.cellsize).astype(np.int32)

        print(
            "Convolving with Gaussian kernel where FWHM {0:2.3f} rad/m^2 - pixels {1}, sigma {2:2.3f} rad/m^2 - pixels {3}"
            .format(rmtf_fwhm, rmtf_fwhm_pixels, sigma_x, sigma_x_pixels)
        )

        clean_beam = Gaussian1DKernel(stddev=sigma_x_pixels)
        clean_beam_array = clean_beam.array

        if x is None:
            q_stokes = sci_signal.convolve(
                self.data.real, clean_beam_array, mode="same", method="fft"
            )
            u_stokes = sci_signal.convolve(
                self.data.imag, clean_beam_array, mode="same", method="fft"
            )
        else:
            q_stokes = sci_signal.convolve(x.real, clean_beam_array, mode="same", method="fft")
            u_stokes = sci_signal.convolve(x.imag, clean_beam_array, mode="same", method="fft")

        p_stokes = q_stokes + 1j * u_stokes
        return p_stokes
