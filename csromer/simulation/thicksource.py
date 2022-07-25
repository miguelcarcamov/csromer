import numpy as np
from scipy.constants import c

from .faradaysource import FaradaySource


class FaradayThickSource(FaradaySource):

    def __init__(self, phi_fg: float = None, phi_center: float = None, **kwargs):
        super().__init__(**kwargs)
        self.phi_fg = phi_fg
        self.phi_center = phi_center
        if phi_center is None:
            self.phi_center = 0.0

    def simulate(self):
        nu = c / np.sqrt(self.lambda2)
        k = (nu / self.nu_0)**(-1.0 * self.spectral_idx)
        const = self.s_nu * k
        # half_phi_fg = self.phi_fg / 2.
        self.data = const * np.exp(2j * (self.lambda2 - self.l2_ref) * self.phi_center
                                   ) * np.sinc(self.phi_fg * (self.lambda2 - self.l2_ref) / np.pi)
