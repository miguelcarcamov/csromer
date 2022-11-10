from dataclasses import dataclass
from typing import Union

import numpy as np

from ...base import Dataset
from .flagger import Flagger

dataclass(init=True, repr=True)


class ManualFlagger(Flagger):
    outlier_idxs: Union[np.ndarray, list, int] = []

    def __post__init__(self):
        super().__post_init__()

    def run(self):
        if isinstance(self.dataset, Dataset):
            original_length = len(self.dataset.sigma)
            sigma_array = self.dataset.sigma.copy()
            if isinstance(self.outlier_idxs, int):
                flagged_percentage = (1.0 / original_length) * 100.0
            else:
                flagged_percentage = (len(self.outlier_idxs) / original_length) * 100.0
            all_idxs = np.arange(0, original_length)
            idxs = np.setxor1d(all_idxs, self.outlier_idxs)
            if self.delete_channels:
                self.dataset.lambda2 = self.dataset.lambda2[idxs]
                self.dataset.sigma = sigma_array[idxs]
                if self.dataset.data is not None:
                    self.dataset.data = self.dataset.data[idxs]
            else:
                self.dataset.w[self.outlier_idxs] = 0.0
            print("Flagging {0:.2f}% of the data".format(flagged_percentage))
            return idxs, self.outlier_idxs
        else:
            raise TypeError("The data attribute is not a Dataset")
