import copy

import numpy as np

from ...objectivefunction import Fi
from ..optimizer import Optimizer


class FixedPointMethod(Optimizer):
    gx: Fi = None

    def run(self):
        n = self.guess_param.n
        xt = self.guess_param.data
        xt1 = np.zeros(n, dtype=xt.dtype)
        e = 1
        iter = 0

        while e > self.tol and iter < self.maxiter:
            xt1 = self.gx(xt)
            e = np.sum(np.abs(xt1 - xt))
            xt = xt1
            iter = iter + 1

        param = copy.deepcopy(self.guess_param)
        param.data = xt1
        return e, param
