from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np

from ..base import Dataset


def median_absolute_deviation(x):
    """
    Returns the median absolute deviation from the window's median
    :param x: Values in the window
    :return: MAD
    """
    return np.median(np.abs(x - np.median(x)))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


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


def normal_flagging(x, mean_sigma=None, nsigma=0.0):
    n = len(x)
    if mean_sigma is None:
        mean_sigma = np.mean(x)
    std_err = np.std(x) / np.sqrt(n)
    threshold = mean_sigma + nsigma * std_err
    preserved_idxs = np.where(x <= threshold)[0]
    outlier_idxs = np.where(x > threshold)[0]
    return preserved_idxs, outlier_idxs


class Flagger(metaclass=ABCMeta):

    def __init__(self, data: Union[Dataset, np.ndarray] = None, delete_channels=None, nsigma=None):
        self.data = data

        if nsigma is None:
            self.nsigma = 0.0
        else:
            self.nsigma = nsigma

        if delete_channels is None:
            self.delete_channels = False
        else:
            self.delete_channels = delete_channels

    @abstractmethod
    def run(self):
        pass


class ManualFlagger(Flagger):

    def __init__(self, **kwargs):
        super(ManualFlagger, self).__init__(**kwargs)

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


class MeanFlagger(Flagger):

    def __init__(self, **kwargs):
        super(MeanFlagger, self).__init__(**kwargs)

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


class HampelFlagger(Flagger):

    def __init__(self, w=3, imputation=False, **kwargs):
        super(HampelFlagger, self).__init__(**kwargs)
        self.w = w
        self.imputation = imputation

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
