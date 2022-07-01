import numpy as np

from .faradaysource import FaradaySource


class FaradayThinSource(FaradaySource):

    def __init__(self, phi_gal=None, dchi=None, **kwargs):
        super().__init__(**kwargs)
        self.phi_gal = phi_gal
        self.dchi = dchi
        if self.dchi is None:
            self.dchi = 0.0

    def simulate(self):
        nu = c / np.sqrt(self.lambda2)
        k = (nu / self.nu_0)**(-1.0 * self.spectral_idx)
        mu_q = np.cos(2.0 * self.phi_gal * (self.lambda2 - self.l2_ref))
        mu_u = np.sin(2.0 * (self.phi_gal * (self.lambda2 - self.l2_ref) + self.dchi))

        # p = np.mean(np.sqrt(mu_q ** 2 + mu_u ** 2))

        self.data = self.s_nu * k * (mu_q + 1j * mu_u)
