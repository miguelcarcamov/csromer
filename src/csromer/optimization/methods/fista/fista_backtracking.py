import copy
from dataclasses import dataclass

from ....optimization.methods.fista.fista_backtracking_algorithm import fista_backtracking_algorithm
from .fista import FISTA


@dataclass(init=True, repr=True)
class BacktrackingFISTA(FISTA):
    eta: float = None

    def run(self):
        ret, x = fista_backtracking_algorithm(
            self.guess_param.data,
            self.F_obj,
            self.fx.evaluate,
            self.fx.calculate_gradient,
            self.gx,
            self.lipschitz_constant,
            self.eta,
            self.max_iter,
            self.guess_param.n,
            self.tol,
            self.verbose,
        )

        param = copy.deepcopy(self.guess_param)
        param.data = x
        return ret, param
