from dataclasses import dataclass

import numpy as np

from .faradaysource import FaradaySource


@dataclass(init=False, repr=True)
class ManualSource(FaradaySource):

    def simulate(self):
        pass
