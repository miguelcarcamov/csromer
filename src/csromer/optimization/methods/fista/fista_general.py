import copy
from dataclasses import dataclass

from ....optimization.methods.fista.fista_general_algorithm import fista_general_algorithm
from .fista import FISTA


@dataclass(init=True, repr=True)
class GeneralFISTA(FISTA):

    def run(self):
        ret, x = fista_general_algorithm(
            self.guess_param.data,
            self.F_obj.evaluate,
            self.fx.calculate_gradient_fista,
            self.gx,
            self.lipschitz_constant,
            self.max_iter,
            self.guess_param.n,
            self.verbose,
        )

        param = copy.deepcopy(self.guess_param)
        param.data = x
        return ret, param
