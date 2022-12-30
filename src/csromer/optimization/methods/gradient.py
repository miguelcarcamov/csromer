import copy
from dataclasses import dataclass

from scipy.optimize import minimize

from ..optimizer import Optimizer


@dataclass(init=True, repr=True)
class GradientBasedMethod(Optimizer):

    method: str = None

    def __post_init__(self):
        if self.method is None:
            self.method = "CG"

        if self.guess_param is None:
            raise ValueError("Guess parameter cannot be Nonetype")

    def run(self):

        ret = minimize(
            fun=self.F_obj.evaluate,
            x0=self.guess_param.data,
            method=self.method,
            jac=self.F_obj.calculate_gradient,
            tol=self.tol,
            options={
                "maxiter": self.maxiter,
                "disp": self.verbose
            },
        )

        param = copy.deepcopy(self.guess_param)
        param.data = ret.x
        return ret.fun, param
