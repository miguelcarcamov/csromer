import numpy as np


def median_absolute_deviation(x):
    """
    Returns the median absolute deviation from the window's median
    :param x: Values in the window
    :return: MAD
    """
    return np.median(np.abs(x - np.median(x)))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class Flagger(metaclass=ABCMeta):
    def __init__(self, dataset: Dataset = None, delete_channels=None, nsigma=None):
        self.dataset = dataset

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


class MeanFlagger(Flagger):
    def __init__(self, **kwargs):
        super(MeanFlagger, self).__init__(**kwargs)

    def run(self, nsigma=0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if self.dataset is None:
            raise ValueError("Cannot flag dataset without a Dataset object")
        else:
            sigma = self.dataset.sigma
            mean_sigma = nsigma * np.mean(sigma)
            std_err = np.std(sigma) / np.sqrt(len(sigma))
            if self.delete_channels:
                idx_channels = np.where(sigma <= mean_sigma + nsigma * std_err)
                self.dataset.sigma = self.dataset.sigma[idx_channels]
                self.dataset.lambda2 = self.dataset.lambda2[idx_channels]
                self.dataset.data = self.dataset.data[idx_channels]
            else:
                flagged_idx = np.where(sigma > mean_sigma + nsigma * std_err)
                self.dataset.w[flagged_idx] = 0.0


class HalperFlagger(Flagger):
    def __init__(self, w=3, imputation=False, **kwargs):
        super(HalperFlagger, self).__init__(**kwargs)
        self.w = w
        self.imputation = imputation

    def run(self, nsigma=0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if self.dataset is None:
            raise ValueError("Cannot flag dataset without a Dataset object")
        else:
            k = 1.4826
            rolling_mean = moving_average(self.dataset.sigma, w)
            rolling_median = np.median(rolling_mean)
            rolling_sigma = k * median_absolute_deviation(rolling_mean)
            if self.imputation:
                if self.delete_channels:
                    idx_channels = np.where(np.abs(self.dataset.sigma - rolling_median) <= (nsigma * rolling_sigma))
                    self.dataset.sigma = self.dataset.sigma[idx_channels]
                else:
                    flagged_idx = np.where(np.abs(self.dataset.sigma - rolling_median) > (nsigma * rolling_sigma))
                    self.dataset.w[flagged_idx] = 0.0
            else:
                flagged_idx = np.where(np.abs(self.dataset.sigma - rolling_median) > (nsigma * rolling_sigma))
                self.dataset.w[flagged_idx] = rolling_median[flagged_idx]



