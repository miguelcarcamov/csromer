import copy
import itertools
import sys
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.constants import c

from ..base.dataset import Dataset


@dataclass(init=False, repr=True)
class FaradaySource(Dataset):
    s_nu: float = None

    def __init__(self, s_nu=None, **kwargs):
        super().__init__(**kwargs)

        self.s_nu = s_nu
        self.sigma = np.ones_like(self.lambda2)

    def __add__(self, other):
        if isinstance(other, FaradaySource) and hasattr(other, "data"):
            if (
                (self.nu == other.nu).all() and self.data is not None and other.data is not None
                and self.s_nu is not None and other.s_nu is not None
            ):
                source_copy = copy.deepcopy(self)
                source_copy.data = self.data + other.data  # Sums the polarized data
                w = self.s_nu + other.s_nu
                source_copy.spectral_idx = (
                    self.s_nu * self.spectral_idx + other.s_nu * other.spectral_idx
                ) / w
                return source_copy
            else:
                raise TypeError("Data attribute in sources cannot be NoneType")

    def __iadd__(self, other):
        if isinstance(other, FaradaySource) and hasattr(other, "data"):
            if (
                (self.nu == other.nu).all() and self.data is not None and other.data is not None
                and self.s_nu is not None and other.s_nu is not None
            ):
                source_copy = copy.copy(self)
                source_copy.data = self.data + other.data  # Sums the polarized data
                w = self.s_nu + other.s_nu
                source_copy.spectral_idx = (
                    self.s_nu * self.spectral_idx + other.s_nu * other.spectral_idx
                ) / w
                return source_copy
            else:
                raise TypeError("Data attribute in sources cannot be NoneType")

    @abstractmethod
    def simulate(self):
        pass

    def add_external_faraday_depolarization(self, sigma_rm=None):
        if sigma_rm is None:
            sigma_rm = 0.0

        self.data *= np.exp(-2.0 * sigma_rm**2 * self.lambda2**2)

    def remove_channels(self, remove_frac=None, random_state=None, chunksize=None):
        if remove_frac == 0.0:
            return
        else:
            remove_frac = 1.0 - remove_frac

        if chunksize is None:
            chunksize = self.m // (10 + (10 * (1.0 - remove_frac)))

        _chansremoved = []
        while True:
            if random_state is None:
                pos = np.random.randint(0, self.m)  # get position
                width = np.random.uniform(0, chunksize)  # get chunk size
            else:
                pos = random_state.randint(0, self.m)  # get position
                width = random_state.uniform(0, chunksize)  # get chunk size
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
            if random_state is None:
                idx = np.random.randint(0, len(chans_removed))
            else:
                idx = random_state.randint(0, len(chans_removed))
            chans_removed = np.delete(chans_removed, idx)
            frac = float(len(chans_removed)) / float(self.m)
            if frac <= remove_frac:
                break

        self.lambda2 = self.lambda2[chans_removed]

        if self.data is not None:
            self.data = self.data[chans_removed]

    def apply_noise(self, noise=None, random_state=None):
        applied_noise = np.zeros((2, ), dtype=np.float32)

        if noise is not None:
            if isinstance(noise, float):
                applied_noise[0] = noise
                applied_noise[1] = noise
            elif isinstance(noise, complex):
                applied_noise[0] = noise.real
                applied_noise[1] = noise.imag
            else:
                raise TypeError("Noise must be either a float or complex number")
            avg_noise = (applied_noise[0] + applied_noise[1]) / 2.
        else:
            return

        if avg_noise > 0.0:
            self.sigma = np.ones_like(self.lambda2) * avg_noise
        else:
            self.sigma = np.ones_like(self.lambda2)
            return

        if random_state is None:
            q_noise = np.random.normal(loc=0.0, scale=applied_noise[0], size=self.m)
            u_noise = np.random.normal(loc=0.0, scale=applied_noise[1], size=self.m)
        else:
            q_noise = random_state.normal(loc=0.0, scale=applied_noise[0], size=self.m)
            u_noise = random_state.normal(loc=0.0, scale=applied_noise[1], size=self.m)
        p_noise = q_noise + 1j * u_noise
        self.data += p_noise
