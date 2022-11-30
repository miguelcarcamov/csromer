from __future__ import annotations

import sys
from abc import ABCMeta
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import astropy.units as u
import numpy as np
import scipy.signal as sci_signal
import scipy.stats
from scipy import special
from scipy.constants import speed_of_light as c

if TYPE_CHECKING:
    from ..transformers.gridding import Gridding


def calculate_sigma(
    image=None,
    x0=0,
    xn=0,
    y0=0,
    yn=0,
    sigma_error=None,
    residual_cal_error=None,
    nbeam=None,
):
    if sigma_error is None and residual_cal_error is None and nbeam is None:
        sigma = np.sqrt(np.mean(image[y0:yn, x0:xn]**2))
    else:
        flux = np.sum(image)
        sigma = np.sqrt((residual_cal_error * flux)**2 + (sigma_error * np.sqrt(nbeam))**2)

    return sigma


def autocorr_gridded(x: np.ndarray):
    variance = x.var()
    x_input = x - x.mean()
    result = sci_signal.correlate(x_input, x_input, mode="full", method="auto")
    result /= variance * len(x)
    return result[result.size // 2:]


def boxpierce(x: np.ndarray = None, k: Union[List, int] = None, conf_level: float = 0.95):
    n = len(x)
    if type(k) == list:
        res = []
        for i in k:
            if i < 1:
                raise ValueError("Cannot calculate for lags lower than 1")
            idx = np.arange(1, i + 1, 1)
            x_sum = n * np.sum(x[idx]**2)
            res.append(x_sum)
        chi2 = scipy.stats.chi2.ppf(conf_level, df=k)
        return np.array(res), chi2
    else:
        if k < 1:
            raise ValueError("The lag cannot be less than 1")
        idx = np.arange(1, k + 1, 1)
        x_sum = n * np.sum(x[idx]**2)
        return np.array(x_sum), scipy.stats.chi2.ppf(conf_level, df=k)


def ljungbox(x: np.ndarray = None, k: Union[List, int] = None, conf_level: float = 0.95):
    n = len(x)
    if isinstance(k, list):
        res = []
        for i in k:
            if i < 1:
                raise ValueError("Cannot calculate for lags lower than 1")
            idx = np.arange(1, i + 1, 1)
            x_sum = n * (n + 2) * np.sum(x[idx]**2 / (n - idx))
            res.append(x_sum)
        chi2 = scipy.stats.chi2.ppf(conf_level, df=k)
        return np.array(res), chi2
    else:
        if k < 1:
            raise ValueError("The lag cannot be less than 1")
        idx = np.arange(1, k + 1, 1)
        x_sum = n * (n + 2) * np.sum(x[idx]**2 / (n - idx))
        return np.array(x_sum), scipy.stats.chi2.ppf(conf_level, df=k)


@dataclass(init=False, repr=True)
class Dataset(metaclass=ABCMeta):
    nu: np.ndarray = None
    lambda2: np.ndarray = None
    data: np.ndarray = None
    l2_ref: float = None
    w: np.ndarray = None
    sigma: np.ndarray = None
    spectral_idx: float = None
    gridded: bool = None
    s: np.ndarray = None
    model_data: np.ndarray = None
    m: int = None
    theo_noise: float = None
    nu_0: float = None
    k: float = None

    def __init__(
        self,
        nu=None,
        lambda2=None,
        data=None,
        l2_ref=None,
        w=None,
        sigma=None,
        spectral_idx=None,
        gridded=None,
    ):

        self.k = None
        self.l2_ref = l2_ref

        if self.l2_ref is None:
            self.l2_ref = 0.0

        self.nu_0 = None
        self.delta_l2_min = 0.0
        self.delta_l2_max = 0.0
        self.delta_l2_mean = 0.0
        self.theo_noise = None
        self.w = None
        self.lambda2 = lambda2
        self.nu = nu
        self.spectral_idx = spectral_idx
        self.gridded = gridded

        if self.gridded is None:
            self.gridded = False

        if self.nu is None and self.lambda2 is None:
            self.s = None

        if lambda2 is not None:
            self.m = len(self.lambda2)
        elif nu is not None:
            self.m = len(self.nu)
        else:
            self.m = None

        if sigma is None and w is None and self.m is not None:
            self.sigma = np.ones(self.m)
        elif w is not None:
            self.w = w
        else:
            self.sigma = sigma

        self.data = data
        self.residual = None
        if self.data is not None:
            self.model_data = np.zeros_like(self.data, dtype=self.data.dtype)
        else:
            self.model_data = None

    @property
    def spectral_idx(self):
        return self.__spectral_idx

    @spectral_idx.setter
    def spectral_idx(self, val):
        if val is None:
            self.__spectral_idx = 0.0
        else:
            self.__spectral_idx = val

        if self.__lambda2 is not None and self.__nu_0 is not None:
            nu = c / np.sqrt(self.__lambda2)
            self.s = (nu / self.__nu_0)**self.__spectral_idx

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, val):
        self.__s = val
        if self.__s is not None:
            self.k = np.sum(self.w / self.__s)

    @property
    def nu_0(self):
        return self.__nu_0

    @nu_0.setter
    def nu_0(self, val):
        self.__nu_0 = val

    @property
    def nu(self):
        return self.__nu

    @nu.setter
    def nu(self, val):
        self.__nu = val
        if val is not None:
            self.__nu_0 = 0.5 * (np.min(val) + np.max(val))
            self.nu_to_l2()

    @property
    def lambda2(self):
        return self.__lambda2

    @lambda2.setter
    def lambda2(self, val):
        self.__lambda2 = val
        if val is not None:
            if all(np.diff(val) < 0):
                val = val[::-1]
                self.__lambda2 = val
            self.__m = len(val)
            self.__nu = c / np.sqrt(val)
            self.__nu_0 = 0.5 * (np.min(self.__nu) + np.max(self.__nu))
            if hasattr(self, "spectral_idx"):
                self.__s = (self.__nu / self.__nu_0)**self.__spectral_idx
            self.w = np.ones(self.__m)
            self.calculate_l2_cellsize()

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, val):
        self.__k = val

    @property
    def m(self):
        return self.__m

    @m.setter
    def m(self, val):
        self.__m = val

    @property
    def theo_noise(self):
        return self.__theo_noise

    @theo_noise.setter
    def theo_noise(self, val):
        self.__theo_noise = val

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val
        if val is not None and isinstance(val, np.ndarray):
            aux_copy = val.copy()
            aux_copy[aux_copy != 0] = 1.0 / np.sqrt(aux_copy[aux_copy != 0])
            self.__sigma = aux_copy
            if hasattr(self, "s"):
                if self.__s is not None:
                    self.k = np.sum(val / self.__s)
            else:
                self.k = np.sum(val)
            if self.__l2_ref is None:
                self.__l2_ref = self.calculate_l2ref()
        self.__theo_noise = self.calculate_theo_noise()

    @property
    def l2_ref(self):
        return self.__l2_ref

    @l2_ref.setter
    def l2_ref(self, val):
        self.__l2_ref = val

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, val):
        if val is not None:
            self.w = 1.0 / (val**2)
        self.__sigma = val

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, val):
        if val is not None:
            if len(val) == self.m:
                self.__data = val
            else:
                self.__m = len(val)
                self.__data = val
            if hasattr(self, "model_data"):
                if self.__model_data is None:
                    self.__model_data = np.zeros_like(val, dtype=self.data.dtype)
        else:
            self.__data = None

    @property
    def model_data(self):
        return self.__model_data

    @model_data.setter
    def model_data(self, val):
        if val is not None:
            if len(val) == self.m:
                self.__model_data = val
                if self.data is not None:
                    self.calculate_residuals()
            else:
                sys.exit("Data must have same size as lambda2")
        else:
            self.__model_data = None

    def nu_to_l2(self):
        lambda2 = (c / self.nu)**2
        self.lambda2 = lambda2[::-1]

    def calculate_amplitude(self, column: "str" = "data"):
        if hasattr(self, column):
            data = getattr(self, column)
            if data.dtype == np.complex64 or data.dtype == np.complex128:
                amplitude = np.abs(data)
                return amplitude
            else:
                raise TypeError("Data is not complex")
        else:
            raise ValueError("Column does not exist")

    def calculate_polangle(self, column: "str" = "data"):
        if hasattr(self, column):
            data = getattr(self, column)
            if data.dtype == np.complex64 or data.dtype == np.complex128:
                pol_angle = 0.5 * np.arctan2(self.data.imag, self.data.real)
                return pol_angle * u.rad
            else:
                raise TypeError("Data is not complex")
        else:
            raise ValueError("Column does not exist")

    def calculate_l2ref(self):
        if self.lambda2 is not None:
            sum_weights = np.sum(self.w)
            return np.sum(self.w * self.lambda2) / sum_weights
        else:
            return None

    def calculate_l2_cellsize(self):
        if self.w is not None:
            lambda2_aux = self.lambda2[self.w > 0.0]
            delta_l2_min = np.min(np.abs(np.diff(lambda2_aux)))
            delta_l2_mean = np.mean(np.abs(np.diff(lambda2_aux)))
            delta_l2_max = np.max(np.abs(np.diff(lambda2_aux)))

            self.delta_l2_min = delta_l2_min
            self.delta_l2_max = delta_l2_max
            self.delta_l2_mean = delta_l2_mean

    def calculate_theo_noise(self):
        if self.w is None:
            return None
        else:
            if isinstance(self.w, np.ndarray):
                if (self.w == 1.0).all():
                    return None
                else:
                    return 1.0 / np.sqrt(np.sum(self.w))
            else:
                return None

    def calculate_residuals(self):
        self.residual = self.data - self.model_data

    def subtract_galacticrm(self, phi_gal):
        p = self.data
        galrm_shift = np.exp(-2j * phi_gal * self.lambda2)
        p_hat = p * galrm_shift
        self.data = p_hat

    def assess_residuals(self, gridding_object: Gridding = None, confidence_interval: float = 0.95):
        if self.gridded:
            autocorr_real = autocorr_gridded(self.residual.real)
            autocorr_imag = autocorr_gridded(self.residual.imag)
            autocorr_real_sq = autocorr_gridded(self.residual.real**2)
            autocorr_imag_sq = autocorr_gridded(self.residual.imag**2)
            lags = sci_signal.correlation_lags(self.m, self.m, mode="full")
        else:
            # Grid the irregular data
            gridding = gridding_object
            gridded_data = gridding.run()
            autocorr_real = autocorr_gridded(gridded_data.residual.real)
            autocorr_imag = autocorr_gridded(gridded_data.residual.imag)
            autocorr_real_sq = autocorr_gridded(gridded_data.residual.real**2)
            autocorr_imag_sq = autocorr_gridded(gridded_data.residual.imag**2)
            lags = sci_signal.correlation_lags(gridded_data.m, gridded_data.m, mode="full")

        autocorr_res = autocorr_real + 1j * autocorr_imag
        autocorr_res_sq = autocorr_real_sq + 1j * autocorr_imag_sq

        lags_pos = np.where(lags >= 0)
        lags = lags[lags_pos]

        vcrit = np.sqrt(2) * special.erfinv(confidence_interval)
        bound = vcrit / np.sqrt(self.m)

        elem_real = ((autocorr_res.real > -bound) & (autocorr_res.real < bound)).sum()
        percentage_real_in = 100.0 * elem_real / len(lags)
        elem_imag = ((autocorr_res.imag > -bound) & (autocorr_res.imag < bound)).sum()
        percentage_imag_in = 100.0 * elem_imag / len(lags)

        elem_real_sq = ((autocorr_res_sq.real > -bound) & (autocorr_res_sq.real < bound)).sum()
        percentage_real_in_sq = 100.0 * elem_real_sq / len(lags)
        elem_imag_sq = ((autocorr_res_sq.imag > -bound) & (autocorr_res_sq.imag < bound)).sum()
        percentage_imag_in_sq = 100.0 * elem_imag_sq / len(lags)

        return (
            lags,
            autocorr_res,
            autocorr_res_sq,
            bound,
            percentage_real_in,
            percentage_imag_in,
            percentage_real_in_sq,
            percentage_imag_in_sq,
        )

    def histogram_residuals(self):
        hist_real, bins_real = np.histogram(self.residual.real, bins="auto")
        hist_imag, bins_imag = np.histogram(self.residual.imag, bins="auto")

        # You can plot the histograms this way:
        # plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")

        return hist_real, bins_real, hist_imag, bins_imag
