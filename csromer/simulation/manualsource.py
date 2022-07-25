import numpy as np

from .faradaysource import FaradaySource


class ManualSource(FaradaySource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def simulate(self):
        pass
