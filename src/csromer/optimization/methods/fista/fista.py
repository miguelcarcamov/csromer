import copy
from dataclasses import dataclass

from ....objectivefunction import Chi2, Fi
from ....optimization.optimizer import Optimizer


@dataclass(init=True, repr=True)
class FISTA(Optimizer):

    fx: Chi2 = None
    gx: Fi = None
    lipschitz_constant: float = None

    def run(self):
        pass
