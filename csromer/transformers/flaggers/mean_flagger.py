from dataclasses import dataclass
from typing import Union

import numpy as np

from ...base import Dataset
from .flagger import Flagger


def normal_flagging(x, mean_sigma=None, nsigma=0.0):
    n = len(x)
    if mean_sigma is None:
        mean_sigma = np.mean(x)
    std_err = np.std(x) / np.sqrt(n)
    threshold = mean_sigma + nsigma * std_err
    preserved_idxs = np.where(x <= threshold)[0]
    outlier_idxs = np.where(x > threshold)[0]
    return preserved_idxs, outlier_idxs


@dataclass(init=True, repr=True)
class MeanFlagger(Flagger):

    def run(self, mean_sigma: Union[np.ndarray, float] = None, nsigma: float = 0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if isinstance(self.data, Dataset):
            original_length = len(self.data.sigma)
            sigma_array = self.data.sigma.copy()
            idxs, outlier_idxs = normal_flagging(sigma_array, mean_sigma, nsigma)
            flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
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
            idxs, outlier_idxs = normal_flagging(self.data, nsigma)
            flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
            if self.delete_channels:
                self.data = self.data[idxs]
            else:
                self.data[outlier_idxs] = 0.0
            print("Flagging {0:.2f}% of the data".format(flagged_percentage))
            return idxs, outlier_idxs
        else:
            raise TypeError("The data attribute is not a Dataset or numpy array object")
