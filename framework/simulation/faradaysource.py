from abc import ABCMeta, abstractmethod
from ..base.dataset import Dataset
import numpy as np
from scipy.constants import c
import itertools
import sys


class FaradaySource(Dataset, metaclass=ABCMeta):
    def __init__(self, nu_0=None, s_nu=None, remove_frac=None, noise=None, spectral_idx=None, **kwargs):
        super().__init__(**kwargs)
        # self.dataset = Dataset(nu=nu)

        if nu_0 is None and self.nu is not None:
            self.nu_0 = np.median(self.nu)
        else:
            self.nu_0 = nu_0

        self.s_nu = s_nu
        self.remove_frac = remove_frac
        self.noise = noise
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

    @abstractmethod
    def simulate(self):
        pass

    def remove_channels(self, remove_frac=None):
        if remove_frac is None:
            remove_frac = 1.0 - self.remove_frac
        elif remove_frac == 0.0:
            return
        else:
            remove_frac = 1.0 - remove_frac

        _chansremoved = []
        while True:
            pos = np.random.randint(0, self.m)  # get position
            width = np.random.uniform(0, 100)  # get chunk size
            low = int(pos - 0.5 * width)
            if low < 0:
                low = 0
            high = int(pos + 0.5 * width)
            if high >= self.m:
                high = self.m - 1

            _chansremoved.append(list(np.arange(low, high)))
            merged = list(itertools.chain(*_chansremoved))

            chans_removed = np.unique(np.ravel(np.array(merged)))
            frac = float(len(chans_removed)) / float(self.m)
            if frac > remove_frac:
                break

        # adjust back towards specified fraction
        # using single channel adjustments:
        while True:
            idx = np.random.randint(0, len(chans_removed))
            chans_removed = np.delete(chans_removed, idx)
            frac = float(len(chans_removed)) / float(self.m)
            if frac <= remove_frac:
                break

        self.lambda2 = self.lambda2[chans_removed]

        if self.data is not None:
            self.data = self.data[chans_removed]

    def apply_noise(self, noise=None):
        if noise is None:
            noise = self.noise
        q_noise = np.random.normal(loc=0.0, scale=noise, size=self.m)
        u_noise = np.random.normal(loc=0.0, scale=noise, size=self.m)
        p_noise = q_noise + 1j * u_noise
        self.data += p_noise


class FaradayThinSource(FaradaySource):
    def __init__(self, phi_gal=None, **kwargs):
        super().__init__(**kwargs)
        self.phi_gal = phi_gal

    def simulate(self):
        nu = c / np.sqrt(self.lambda2)
        k = (nu / self.nu_0) ** (-1.0 * self.spectral_idx)
        mu_q = np.cos(2. * self.phi_gal * (self.lambda2 - self.l2_ref))
        mu_u = np.sin(2. * self.phi_gal * (self.lambda2 - self.l2_ref))

        # p = np.mean(np.sqrt(mu_q ** 2 + mu_u ** 2))

        self.data = self.s_nu * k * (mu_q + 1j * mu_u)


class FaradayThickSource(FaradaySource):
    def __init__(self, phi_fg: float = None, phi_center: float = None, **kwargs):
        super().__init__(**kwargs)
        self.phi_fg = phi_fg
        self.phi_center = phi_center
        if phi_center is None:
            self.phi_center = 0.0

    def simulate(self):
        # nu = c / np.sqrt(self.lambda2)
        # k = (nu / self.nu_0) ** (-1.0 * self.spectral_idx)
        phi_fg = self.phi_fg / 2.
        j = 2. * (self.lambda2 - self.l2_ref) * (self.phi_center + phi_fg)
        k = 2. * (self.lambda2 - self.l2_ref) * (self.phi_center - phi_fg)
        mu_q = np.sin(j) - np.sin(k)
        const = self.s_nu / (2. * self.phi_fg * (self.lambda2 - self.l2_ref))
        mu_u = np.cos(j) - np.cos(k)

        self.data = const * (mu_q + mu_u / 1j)
