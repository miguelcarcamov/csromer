from dataclasses import dataclass

import numpy as np

from ..fi import Fi


def approx_abs(x, epsilon):
    return np.sqrt(x * x + epsilon)


@dataclass(init=True, repr=True)
class L1(Fi):

    def __post_init__(self):
        super().__post_init__()

    def evaluate(self, x, epsilon=np.finfo(np.float32).tiny):
        val = np.sum(approx_abs(x, epsilon))
        # print("Evaluation on L1:", val)
        return val

    def calculate_gradient(self, x, epsilon=np.finfo(np.float32).tiny):
        dx = np.zeros(len(x), x.dtype)

        dx = x / approx_abs(x, epsilon)

        return dx

    def calculate_prox(self, x, nu=0):
        l1_prox = np.sign(x) * np.maximum(np.abs(x) - self.reg, 0.0)
        return l1_prox
