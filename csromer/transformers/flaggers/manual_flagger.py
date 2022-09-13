from dataclasses import dataclass
from typing import Union

import numpy as np

from ...base import Dataset
from .flagger import Flagger

dataclass(init=True, repr=True)


class ManualFlagger(Flagger):

    def run(self, outlier_idxs: Union[np.ndarray, list, int]):
        if isinstance(self.data, Dataset):
            original_length = len(self.data.sigma)
            sigma_array = self.data.sigma.copy()
            if isinstance(outlier_idxs, int):
                flagged_percentage = (1.0 / original_length) * 100.0
            else:
                flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
            all_idxs = np.arange(0, original_length)
            idxs = np.setxor1d(all_idxs, outlier_idxs)
            if self.delete_channels:
                self.data.lambda2 = self.data.lambda2[idxs]
                self.data.sigma = sigma_array[idxs]
                if self.data.data is not None:
                    self.data.data = self.data.data[idxs]
            else:
                self.data.w[outlier_idxs] = 0.0
            print("Flagging {0:.2f}% of the data".format(flagged_percentage))
            return idxs, outlier_idxs
        elif isinstance(self.data, np.ndarray):
            original_length = len(self.data)
            all_idxs = np.arange(0, original_length)

            if isinstance(outlier_idxs, int):
                flagged_percentage = (1.0 / original_length) * 100.0
            else:
                flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
            idxs = np.setxor1d(all_idxs, outlier_idxs)

            if self.delete_channels:
                self.data = self.data[idxs]
            else:
                self.data[outlier_idxs] = 0.0
            print("Flagging {0:.2f}% of the data".format(flagged_percentage))
            return idxs, outlier_idxs
        else:
            raise TypeError("The data attribute is not a Dataset or numpy array object")
