from abc import ABCMeta, abstractmethod
from ..base.dataset import Dataset
import numpy as np


class FaradaySource(metaclass=ABCMeta):
    def __init__(self, nu=None, nu_0=None, s_nu=None, remove_frac=None, phi_gal=None, noise=None):
        self.dataset = Dataset()
        if nu is not None:
            dataset.nu = nu

        self.nu_0 = nu_0
        self.s_nu = s_nu
        self.remove_frac = remove_frac
        self.phi_gal = phi_gal
        self.noise = noise

    @abstractmethod
    def simulate(self):
        pass

    def remove_channels(self):
        _chansremoved = []
        while True:
            pos = np.random.randint(0, dataset.m)  # get position
            width = np.random.uniform(0, 100)  # get chunk size
            low = int(pos - 0.5 * width)
            if low < 0:
                low = 0
            high = int(pos + 0.5 * width)
            if high >= dataset.m:
                high = dataset.m - 1

            _chansremoved.append(list(np.arange(low, high)))
            merged = list(itertools.chain(*_chansremoved))

            chans_removed = np.unique(np.ravel(np.array(merged)))
            frac = float(len(chans_removed)) / float(dataset.m)
            if frac > self.remove_frac:
                break

        # adjust back towards specified fraction
        # using single channel adjustments:
        while True:
            idx = np.random.randint(0, len(chans_removed))
            chans_removed = np.delete(chans_removed, idx)
            frac = float(len(chans_removed)) / float(dataset.m)
            if frac <= remove_frac:
                break

        self.dataset.nu = self.dataset.nu[chans_removed]

    def apply_noise(self):
        q_noise = np.random.normal(loc=0.0, scale=self.noise, size=len(self.dataset.data))
        u_noise = np.random.normal(loc=0.0, scale=self.noise, size=len(self.dataset.data))
        p_noise = q_noise + 1j * u_noise
        self.dataset.data += p_noise


class FaradayThinSource(FaradaySource):
    def __init__(self, spectral_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.spectral_idx = spectral_idx

    @property
    def spectral_idx(self):
        return self.__spectral_idx

    @spectral_idx.setter
    def spectral_idx(self, val):
        if val is None:
            self.__spectral_idx = 0.0
        else:
            self.__spectral_idx = val

    def simulate(self):
        k = (self.dataset.nu / self.nu_0) ** (-1.0 * self.spectral_idx)
        mu_q = self.s_nu * k * np.cos(2. * self.phi_gal * self.dataset.lambda2)
        mu_u = self.s_nu * k * np.sin(2. * self.phi_gal * self.dataset.lambda2)

        # p = np.mean(np.sqrt(mu_q ** 2 + mu_u ** 2))

        self.dataset.data = mu_q + 1j * mu_u


class FaradayThickSource(FaradaySource):
    def __init__(self, phi_fg=None, **kwargs):
        super().__init__(**kwargs)
        self.phi_fg = phi_fg

    def simulate(self):
        mu_q = self.s_nu * np.sin(2 * self.phi_fg * self.dataset.lambda2) / (2. * self.phi_fg * self.dataset.lambda2)
        mu_u = 0.0

        self.dataset.data = mu_q + 1j * mu_u

