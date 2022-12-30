from dataclasses import dataclass

import numpy as np

from ..fi import Fi


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


def approx_l2_norm(x, epsilon):
    return l2_norm(x + epsilon)


@dataclass(init=True, repr=True)
class L2(Fi):

    def __post_init__(self):
        super().__post_init__()

    def evaluate(self, x, epsilon=np.finfo(np.float32).tiny):
        val = approx_l2_norm(x, epsilon)
        return val

    def calculate_gradient(self, x, epsilon=np.finfo(np.float32).tiny):
        dx = np.zeros(len(x), x.dtype)

        dx = x / approx_l2_norm(x, epsilon)

        return dx

    def calculate_prox(self, x, nu=0):
        l2_factor = 1. - (self.reg / l2_norm(x))
        l2_prox = np.maximum(l2_factor, 0.0)
        print(l2_prox)
        return x * l2_prox
