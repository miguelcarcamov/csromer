from scipy.constants import speed_of_light as c
import sys
import numpy as np
import scipy.signal as sci_signal
from scipy import special

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

        if nu is not None:
            self.nu = nu
        elif lambda2 is not None:
            self.lambda2 = lambda2

        if sigma is None and w is None:
            self.w = np.ones(self.m)
        elif w is not None:
            self.w = w
        else:
            self.sigma = sigma

        self.m = len(self.lambda2)

        self.data = data
        self.residual = None
        if self.data is not None:
            self.model_data = np.zeros_like(self.data, dtype=self.data.dtype)
        else:
            self.model_data = None

    @property
    def nu(self):
        return self.__nu

    @nu.setter
    def nu(self, val):
        self.__nu = val
        self.nu_to_l2()

    @property
    def lambda2(self):
        return self.__lambda2

    @lambda2.setter
    def lambda2(self, val):
        self.__lambda2 = val
        self.__m = len(val)
        self.calculate_l2_cellsize()

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val
        self.k = np.sum(self.__w)
        self.__l2_ref = self.calculate_l2ref()

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
                sys.exit("Data must have same size as lambda2")
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
                    self.calculate_residual()
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
            sys.exit("Could not calculate l2 reference, lambda2 is defined as None")

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

    def calculate_residual(self):
        self.residual = self.model_data - self.data

    def assess_residual(self):
        autocorr_real = autocorr(self.residual.real)
        autocorr_imag = autocorr(self.residual.imag)
        autocorr_res = autocorr_real + 1j * autocorr_imag

        lags = sci_signal.correlation_lags(self.m, self.m, mode="full")
        lags_pos = np.where(lags >= 0)
        lags = lags[lags_pos]

        vcrit = np.sqrt(2) * special.erfinv(0.95)
        bound = vcrit / np.sqrt(self.m)

        elem_real = ((autocorr_res.real > -bound) & (autocorr_res.real < bound)).sum()
        percentage_real_in = 100.0 * elem_real / len(lags)
        elem_imag = ((autocorr_res.imag > -bound) & (autocorr_res.imag < bound)).sum()
        percentage_imag_in= 100.0 * elem_imag / len(lags)

        return lags, autocorr_res, bound, percentage_real_in, percentage_imag_in




