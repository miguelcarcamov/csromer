from scipy.constants import speed_of_light as c
import sys
import numpy as np
import scipy.signal as sci_signal
import astropy.units as u
from scipy import special
import copy


def calculate_sigma(image=None, x0=0, xn=0, y0=0, yn=0, sigma_error=None, residual_cal_error=None,
                    nbeam=None):
    if sigma_error is None and residual_cal_error is None and nbeam is None:
        sigma = np.sqrt(np.mean(image[y0:yn, x0:xn] ** 2))
    else:
        flux = np.sum(image)
        sigma = np.sqrt((residual_cal_error * flux) ** 2 + (sigma_error * np.sqrt(nbeam)) ** 2)

    return sigma


def autocorr(x):
    variance = x.var()
    x_input = x - x.mean()
    result = sci_signal.correlate(x_input, x_input, mode='full', method='auto')
    result /= variance*len(x)
    return result[result.size // 2:]


class Dataset:
    def __init__(self, nu=None, lambda2=None, data=None, w=None, sigma=None, spectral_idx=None):
        self.k = None
        self.l2_ref = None
        self.nu_0 = None
        self.delta_l2_min = 0.0
        self.delta_l2_max = 0.0
        self.delta_l2_mean = 0.0
        self.w = None
        self.lambda2 = lambda2
        self.nu = nu
        self.spectral_idx = spectral_idx
        if self.nu is None:
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
            self.__s = (nu / self.__nu_0) ** (-1.0 * self.__spectral_idx)

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, val):
        self.__s = val

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
            self.__nu_0 = np.median(val)
            self.nu_to_l2()

    @property
    def lambda2(self):
        return self.__lambda2

    @lambda2.setter
    def lambda2(self, val):
        self.__lambda2 = val
        if val is not None:
            self.__m = len(val)
            self.__nu = c / np.sqrt(val)
            self.__nu_0 = np.median(self.__nu)
            if hasattr(self, 'spectral_idx'):
                self.__s = (self.__nu / self.__nu_0) ** (-1.0 * self.__spectral_idx)
            if self.__w is not None:
                if len(val) != len(self.__w):
                    self.w = np.ones(self.__m)
            else:
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
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val
        if val is not None:
            aux_copy = val.copy()
            aux_copy[aux_copy != 0] = 1.0 / np.sqrt(aux_copy[aux_copy != 0])
            self.__sigma = aux_copy
            self.__k = np.sum(val)
            self.__l2_ref = self.calculate_l2ref()

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
            self.w = 1.0 / (val ** 2)
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
            if hasattr(self, 'model_data'):
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
        lambda2 = (c / self.nu) ** 2
        self.lambda2 = lambda2[::-1]

    def calculate_amplitude(self, column: 'str' = 'data'):
        if hasattr(self, column):
            data = getattr(self, column)
            if data.dtype == np.complex64 or data.dtype == np.complex128:
                amplitude = np.abs(data)
                return amplitude
            else:
                raise TypeError("Data is not complex")
        else:
            raise ValueError("Column does not exist")

    def calculate_polangle(self, column: 'str' = 'data'):
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
            return (1. / self.k) * np.sum(self.w * self.lambda2)
        else:
            return None

    def calculate_l2_cellsize(self):

        delta_l2_min = np.min(np.abs(np.diff(self.lambda2)))
        delta_l2_mean = np.mean(np.abs(np.diff(self.lambda2)))
        delta_l2_max = np.max(np.abs(np.diff(self.lambda2)))

        self.delta_l2_min = delta_l2_min
        self.delta_l2_max = delta_l2_max
        self.delta_l2_mean = delta_l2_mean

    # The next functions need to be checked again !!!
    def calculate_sigmas_cube(self, image=None, x0=0, xn=0, y0=0, yn=0):
        sigma = np.zeros(len(self.nu))

        for i in range(len(self.nu)):
            sigma[i] = np.sqrt(np.mean(image[i, y0:yn, x0:xn] ** 2))

        self.sigma = sigma

    def calculate_residuals(self):
        self.residual = self.data - self.model_data

    def assess_residuals(self, confidence_interval=0.95):
        autocorr_real = autocorr(self.residual.real)
        autocorr_imag = autocorr(self.residual.imag)
        autocorr_real_sq = autocorr(self.residual.real**2)
        autocorr_imag_sq = autocorr(self.residual.imag**2)
        autocorr_res = autocorr_real + 1j * autocorr_imag
        autocorr_res_sq = autocorr_real_sq + 1j * autocorr_imag_sq

        lags = sci_signal.correlation_lags(self.m, self.m, mode="full")
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

        return lags, autocorr_res, autocorr_res_sq, bound, percentage_real_in, percentage_imag_in, percentage_real_in_sq, percentage_imag_in_sq

    def histogram_residuals(self):
        hist_real, bins_real = np.histogram(self.residual.real, bins='auto')
        hist_imag, bins_imag = np.histogram(self.residual.imag, bins='auto')

        # You can plot the histograms this way:
        # plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")

        return hist_real, bins_real, hist_imag, bins_imag
