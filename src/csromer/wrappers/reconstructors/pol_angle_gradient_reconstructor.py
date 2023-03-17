from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .faraday_reconstructor import FaradayReconstructorWrapper


@dataclass
class PolAngleGradientReconstructorWrapper(FaradayReconstructorWrapper):
    initial_phi_0: float = None
    reconstructed_phi_0: float = field(init=False)
    reconstructed_y_value: float = field(init=False)

    def __post_init__(self):
        if self.initial_phi_0 is None:
            self.initial_phi_0 = 0.0

    @staticmethod
    def __line(p, x):
        m, c = p
        return m * x + c

    @staticmethod
    def __dtheta(a, b):
        x1 = np.abs(a - b)
        x2 = 2 * np.pi - np.abs(a - b)
        x = np.vstack((x1, x2))
        return np.amin(x, axis=0)

    def __fit_chi(self, x, y, err, p0):
        nll = lambda *args: np.sum(self.__dtheta(y, self.__line(*args))**2 / err**2)

        initial = p0 + 0.01 * np.random.randn(2)
        bnds = ((None, None), (-np.pi, np.pi))

        soln = minimize(nll, initial, bounds=bnds, args=x)
        p = soln.x

        return p

    def config_fd_space(self):
        pass

    def reconstruct(self):
        # Dataset object calculates the polarization angle as 0.5 * arctan(U/Q)
        pol_angle = 2.0 * self.dataset.calculate_polangle().value
        pfit = 0.5 * self.__fit_chi(
            self.dataset.lambda2, pol_angle, self.dataset.sigma, [self.initial_phi_0, 0.0]
        )
        self.reconstructed_phi_0 = pfit[0]
        self.reconstructed_y_value = pfit[1]
        return pfit

    def calculate_second_moment(self):
        pass
