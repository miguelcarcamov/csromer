from scipy.constants import speed_of_light as c
import sys
import numpy as np
import scipy.signal as sci_signal
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
    result = sci_signal.correlate(x / np.std(x), x / np.std(x), mode='full', method='auto')
    result /= len(x)
    return result[result.size // 2:]


class Dataset:
    def __init__(self, nu=None, lambda2=None, data=None, w=None, sigma=None):
        self.k = None
        self.l2_ref = 0.0
        self.delta_l2_min = 0.0
        self.delta_l2_max = 0.0
        self.delta_l2_mean = 0.0
        self.w = None
        self.lambda2 = lambda2
        self.nu = nu

        if lambda2 is not None:
            self.m = len(self.lambda2)
        elif nu is not None:
            self.m = len(self.nu)
        else:
            self.m = None

        if sigma is None and w is None:
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

    def __add__(self, other):
        if isinstance(other, Dataset) and hasattr(other, 'data'):
            if (self.nu == other.nu).all() and self.data is not None and other.data is not None:
                source_copy = copy.deepcopy(self)
                source_copy.data = self.data + other.data
                return source_copy
            else:
                raise TypeError("Data in sources cannot have NoneType data")

    @property
    def nu(self):
        return self.__nu

    @nu.setter
    def nu(self, val):
        self.__nu = val
        if val is not None:
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
        self.__sigma = val
        self.w = 1.0 / (val ** 2)

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
        self.residual = self.model_data - self.data

    def assess_residuals(self, confidence_interval=0.95):
        autocorr_real = autocorr(self.residual.real)
        autocorr_imag = autocorr(self.residual.imag)
        autocorr_res = autocorr_real + 1j * autocorr_imag

        lags = sci_signal.correlation_lags(self.m, self.m, mode="full")
        lags_pos = np.where(lags >= 0)
        lags = lags[lags_pos]

        vcrit = np.sqrt(2) * special.erfinv(confidence_interval)
        bound = vcrit / np.sqrt(self.m)

        elem_real = ((autocorr_res.real > -bound) & (autocorr_res.real < bound)).sum()
        percentage_real_in = 100.0 * elem_real / len(lags)
        elem_imag = ((autocorr_res.imag > -bound) & (autocorr_res.imag < bound)).sum()
        percentage_imag_in = 100.0 * elem_imag / len(lags)

        return lags, autocorr_res, bound, percentage_real_in, percentage_imag_in

    def histogram_residuals(self):
        hist_real, bins_real = np.histogram(self.residual.real, bins='auto')
        hist_imag, bins_imag = np.histogram(self.residual.imag, bins='auto')

        # You can plot the histograms this way:
        # plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")

        return hist_real, bins_real, hist_imag, bins_imag
