from dataclasses import dataclass

import numpy as np

from ...base import Dataset
from .flagger import Flagger, median_absolute_deviation, moving_average


def hampel(array, w, nsigma, imputation=False):
    k = 1.4826
    array_copy = array.copy()
    rolling_mean = moving_average(array_copy, w)
    rolling_median = np.median(rolling_mean)
    rolling_sigma = k * median_absolute_deviation(rolling_mean)
    preserved_idxs = np.where(np.abs(array_copy - rolling_median) <= (nsigma * rolling_sigma))[0]
    outlier_idxs = np.where(np.abs(array_copy - rolling_median) > (nsigma * rolling_sigma))[0]
    if imputation:
        array_copy[outlier_idxs] = rolling_median
        return array_copy, preserved_idxs, outlier_idxs
    else:
        return preserved_idxs, outlier_idxs


@dataclass(init=True, repr=True)
class HampelFlagger(Flagger):
    w: int = None
    imputation: bool = None

    def run(self, nsigma: float = 0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if isinstance(self.data, Dataset):
            original_length = len(self.data.sigma)
            if self.imputation:
                new_sigma, idxs, outlier_idxs = hampel(
                    self.data.sigma, self.w, nsigma, self.imputation
                )
                self.data.sigma = new_sigma
                flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
                print("Imputing {0:.2f}% of the data".format(flagged_percentage))
                return None
            else:
                sigma_array = self.data.sigma.copy()
                idxs, outlier_idxs = hampel(sigma_array, self.w, nsigma, self.imputation)
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
            if self.imputation:
                new_sigma, idxs, outlier_idxs = hampel(
                    self.data.sigma, self.w, nsigma, self.imputation
                )
                self.data = new_sigma
                flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
                print("Imputing {0:.2f}% of the data".format(flagged_percentage))
                return None
            else:
                idxs, outlier_idxs = hampel(self.data.sigma, self.w, nsigma, self.imputation)
                if self.delete_channels:
                    self.data = self.data[idxs]
                else:
                    self.data[outlier_idxs] = 0.0
                flagged_percentage = (len(outlier_idxs) / original_length) * 100.0
                print("Flagging {0:.2f}% of the data".format(flagged_percentage))
                return idxs, outlier_idxs
        else:
            raise TypeError("The data attribute is not a Dataset or numpy array object")
