import numpy as np
from pytest import fixture


@fixture
def meerkat_dataset():
    nu = np.linspace(start=0.9e9, stop=1.67e9, num=1000)
    yield nu


@fixture
def jvla_dataset():
    nu = np.linspace(start=1.008e9, stop=2.031e9, num=1000)
    yield nu
