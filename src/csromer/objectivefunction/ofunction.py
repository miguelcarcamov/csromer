from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


@dataclass(init=True, repr=True)
class OFunction:
    F: Union[List[object], np.ndarray] = None
    n_funcs: int = 0
    prox_functions: List = field(init=False, default_factory=list)
    values: np.ndarray = field(init=False, default=np.ndarray)

    def __post_init__(self):

        if self.F is None:
            self.prox_functions = []
            self.nfuncs = 0
        else:
            self.values = np.zeros(len(self.F))
            self.nfuncs = len(self.F)
            self.prox_functions = [f_i for f_i in self.F]

    def get_lambda(self, _id=0):
        return self.F[_id].reg

    def set_lambda(self, reg=0.0, _id=0):
        self.F[_id].reg = reg

    def evaluate(self, x):
        ret = 0.0

        for i in range(0, len(self.F)):
            self.values[i] = self.F[i].evaluate(x)
            ret += self.F[i].reg * self.values[i]
        return ret

    def calculate_gradient(self, x):
        res = np.zeros(len(x), dtype=x.dtype)
        for f_i in self.F:
            res += f_i.reg * f_i.calculate_gradient(x)
        return res

    def calc_prox(self, x, nu=0, _id=0):
        if self.nfuncs == 1:
            f_i = self.F[_id]
            proximal = f_i.calculate_prox(x, nu)
        else:
            proximal = x
            for i in range(len(self.prox_functions)):
                proximal = self.F[i].calculate_prox(proximal)
        return proximal
