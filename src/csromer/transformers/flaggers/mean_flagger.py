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

    def __post__init__(self):
        super().__post_init__()

    def run(self, mean_sigma: Union[np.ndarray, float] = None, nsigma: float = 0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if isinstance(self.dataset, Dataset):
            original_length = len(self.dataset.sigma)
            sigma_array = self.dataset.sigma.copy()
            idxs, outlier_idxs = normal_flagging(sigma_array, mean_sigma, nsigma)
            flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
            if self.delete_channels:
                self.dataset.lambda2 = self.dataset.lambda2[idxs]
                self.dataset.sigma = sigma_array[idxs]
                if self.dataset.data is not None:
                    self.dataset.data = self.dataset.data[idxs]
            else:
                self.dataset.w[outlier_idxs] = 0.0
            print("Flagging {0:.2f}% of the data".format(flagged_percentage))
            return idxs, outlier_idxs
        else:
            raise TypeError("The data attribute is not a Dataset")
