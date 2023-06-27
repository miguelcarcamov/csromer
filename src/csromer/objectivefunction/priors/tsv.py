from dataclasses import dataclass, field

import numpy as np
import prox_tv as ptv

from ..fi import Fi


@dataclass(init=True, repr=True)
class TSV(Fi):
    nu: np.ndarray = field(init=False, default=np.array([]))

    def __post_init__(self):
        super().__post_init__()

    def evaluate(self, x):
        n = x.shape[0]
        tmp = x[1:n] - x[0:n - 1]
        tv = np.sum(tmp**2)
        return tv

    def calculate_gradient(self, x):

        n = x.shape[0]
        derivative = np.zeros_like(x)
        idx_plus = np.arange(1, n)
        idx = np.arange(0, n - 1)

        derivative[idx] += x[idx] - x[idx_plus]
        derivative[idx_plus] += x[idx_plus] - x[idx]
        derivative *= 2.0

        return derivative

    def calculate_prox(self, x, nu=0.0):
        return ptv.tv2_1d(x, self.reg)
