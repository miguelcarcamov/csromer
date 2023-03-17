from abc import ABC, ABCMeta, abstractmethod

import numpy as np


class Function1D(metaclass=ABCMeta):

    def __init__(self, x=None):
        if x is not None:
            self.x = x
        else:
            self.x = np.array([])

    @abstractmethod
    def run(self):
        return


class Gaussian(Function1D, ABC):

    def __init__(self, amplitude=None, mu=None, sigma=None, fwhm=None, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        if amplitude is not None:
            self.amplitude = amplitude
        else:
            self.amplitude = 1.0

        if mu is not None:
            self.mu = mu
        else:
            self.mu = 0.0

        if fwhm is None and sigma is not None:
            self.sigma = sigma
        else:
            self.fwhm = fwhm

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, val):
        self.__sigma = val
        val_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        self.__fwhm = val * val_fwhm

    @property
    def fwhm(self):
        return self.__fwhm

    @fwhm.setter
    def fwhm(self, val):
        self.__fwhm = val
        val_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        self.__sigma = val / val_fwhm

    @staticmethod
    def normalize(x, mode="integral"):

        if mode == "integral":
            normalization = x.sum()
        elif mode == "peak":
            normalization = x.max()
        else:
            raise ValueError("invalid mode, must be 'integral' or 'peak'")

        return x / normalization

    def run(self, normalized=True):
        f_gauss = self.amplitude * np.exp(-0.5 * ((self.x - self.mu) / self.sigma)**2)
        if normalized:
            f_gauss = self.normalize(f_gauss)
        return f_gauss
