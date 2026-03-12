from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...base import Dataset
from ...utils.array_utils import asnumpy, math_module
from .flagger import Flagger, median_absolute_deviation, moving_average


def hampel_scalars(sigma: np.ndarray, window: int) -> tuple[float, float]:
    """
    Compute rolling median and rolling sigma (MAD scale) from sigma array.
    Used for Hampel filter; only this part needs numpy (for convolve/median).
    Returns (rolling_median, rolling_sigma).
    """
    k = 1.4826
    rolling_mean = moving_average(np.array(sigma, copy=True), window)
    rolling_median = float(np.median(rolling_mean))
    rolling_sigma = float(k * median_absolute_deviation(rolling_mean))
    return rolling_median, rolling_sigma


@dataclass(init=True, repr=True)
class HampelFlagger(Flagger):
    window: Optional[int] = None
    imputation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.window is None:
            self.window = 5

    def run(self, nsigma: float = 0.0):
        if self.nsigma is not None:
            nsigma = self.nsigma

        if not isinstance(self.dataset, Dataset):
            raise TypeError("The data attribute is not a Dataset")

        sigma = self.dataset.sigma
        window = int(self.window)
        # Only compute sigma for scalar stats (rolling median / MAD)
        sigma_np = asnumpy(sigma)
        original_length = len(sigma_np)
        rolling_median, rolling_sigma = hampel_scalars(sigma_np, window)

        xp = math_module(sigma)
        # Mask: True = keep, False = outlier (dask-friendly)
        mask = xp.abs(sigma - rolling_median) <= (nsigma * rolling_sigma)

        if self.imputation:
            self.dataset.sigma = xp.where(mask, sigma, rolling_median)
            s = xp.sum(mask)
            outlier_count = original_length - int(s.compute() if hasattr(s, "compute") else int(s))
            flagged_percentage = (outlier_count / original_length) * 100.0
            print("Imputing {0:.2f}% of the data".format(flagged_percentage))
            return None
        else:
            # Indices only needed for delete_channels or return value
            mask_np = asnumpy(mask)
            kept_idxs = np.where(mask_np)[0]
            outlier_idxs = np.where(~mask_np)[0]
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
