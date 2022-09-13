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
        tv = 0.0
        n = x.shape[0]
        for i in range(0, n - 1):
            tv += np.abs(x[i + 1] - x[i])**2
        return tv

    def calculate_gradient(self, x):

        n = len(x)
        dx = np.zeros(n, x.dtype)
        for i in range(1, n - 1):
            dx[i] = 2.0 * (np.sign(x[i] - x[i - 1]) - np.sign(x[i + 1] - x[i]))
        return dx

    def calculate_prox(self, x, nu=0.0):
        return ptv.tv2_1d(x, self.reg)
