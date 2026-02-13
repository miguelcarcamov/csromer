from dataclasses import dataclass
from typing import Union

import numpy as np

from ...base import Dataset
from ...utils.array_utils import asnumpy, math_module
from .flagger import Flagger


def mean_flagger_threshold(sigma: np.ndarray, mean_sigma: Union[float, None] = None, nsigma: float = 0.0) -> float:
    """Compute threshold from sigma (numpy); returns scalar threshold."""
    n = len(sigma)
    if mean_sigma is None:
        mean_sigma = float(np.mean(sigma))
    std_err = np.std(sigma) / np.sqrt(n)
    return mean_sigma + nsigma * std_err


@dataclass(init=True, repr=True)
class MeanFlagger(Flagger):

    def __post_init__(self):
        super().__post_init__()

    def run(self, mean_sigma: Union[np.ndarray, float] = None, nsigma: float = 0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if not isinstance(self.dataset, Dataset):
            raise TypeError("The data attribute is not a Dataset")

        sigma = self.dataset.sigma
        sigma_np = asnumpy(sigma)
        original_length = len(sigma_np)
        threshold = mean_flagger_threshold(sigma_np, mean_sigma, nsigma)

        xp = math_module(sigma)
        mask = sigma <= threshold
        kept_idxs = np.where(sigma_np <= threshold)[0]
        outlier_idxs = np.where(sigma_np > threshold)[0]
        flagged_percentage = (len(outlier_idxs) / original_length) * 100.0

        if self.delete_channels:
            self.dataset.lambda2 = self.dataset.lambda2[kept_idxs]
            self.dataset.sigma = self.dataset.sigma[kept_idxs]
            if self.dataset.data is not None:
                self.dataset.data = self.dataset.data[kept_idxs]
            self.dataset.w = self.dataset.w[kept_idxs]
        else:
            self.dataset.w = xp.where(mask, self.dataset.w, xp.zeros_like(self.dataset.w))

        print("Flagging {0:.2f}% of the data".format(flagged_percentage))
        return kept_idxs, outlier_idxs
