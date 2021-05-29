import numpy as np
from abc import ABCMeta, abstractmethod, ABC


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
            val = 2.0 * np.sqrt(2.0 * np.log(2.0))
            self.sigma = fwhm / val

    def run(self):
        return self.amplitude * np.exp(-0.5*((self.x - self.mu) / self.sigma) ** 2)