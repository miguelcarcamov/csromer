import copy
from dataclasses import dataclass

from ....objectivefunction import Chi2, Fi
from ....optimization.optimizer import Optimizer


@dataclass(init=True, repr=True)
class FISTA(Optimizer):

    fx: Chi2 = None
    gx: Fi = None
    lipschitz_constant: float = None

    def __post_init__(self):
        if self.lipschitz_constant is None:
            self.lipschitz_constant = 1.0

    def run(self):
        pass
