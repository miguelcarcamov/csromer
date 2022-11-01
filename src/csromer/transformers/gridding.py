from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..base import Dataset

EPSILON = 1e-12


def complex_bincount(x: np.ndarray = None, complex_array: np.ndarray = None):
    if (
        x is not None and complex_array is not None
        and (complex_array.dtype == np.complex64 or complex_array.dtype == np.complex128)
    ):
        real_part = complex_array.real
        imag_part = complex_array.imag
        bincount_real = np.bincount(x, real_part)
        bincount_imag = np.bincount(x, imag_part)
        bincount_complex = bincount_real + 1j * bincount_imag
        return bincount_complex
    else:
        raise TypeError("Array must be complex and not None")


class Gridding:

    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset

    def run(self):
        gridded_dataset = copy.deepcopy(self.dataset)
        gridded_dataset.gridded = True
        l2_grid = np.arange(
            start=0.0 + EPSILON,
            stop=np.max(self.dataset.lambda2),
            step=self.dataset.delta_l2_mean,
        )
        l2_grid_pos = np.floor(self.dataset.lambda2 // self.dataset.delta_l2_mean).astype(int)
        bincount_data = complex_bincount(l2_grid_pos, self.dataset.w * self.dataset.data)
        bincount_model = complex_bincount(l2_grid_pos, self.dataset.w * self.dataset.model_data)
        bincount_weights = np.bincount(l2_grid_pos, self.dataset.w)
        unique_idx = np.unique(l2_grid_pos)

        m_grid = len(l2_grid)
        gridded_data = np.zeros(m_grid, dtype=np.complex64)
        gridded_model = np.zeros(m_grid, dtype=np.complex64)
        gridded_w = np.zeros(m_grid, dtype=np.float32)

        gridded_data[unique_idx] = bincount_data[unique_idx]
        gridded_model[unique_idx] = bincount_model[unique_idx]
        gridded_w[unique_idx] = bincount_weights[unique_idx]

        valid_idx = np.where(gridded_w > 0.0)
        gridded_data[valid_idx] /= gridded_w[valid_idx]
        gridded_model[valid_idx] /= gridded_w[valid_idx]

        gridded_dataset.lambda2 = l2_grid
        gridded_dataset.w = gridded_w
        gridded_dataset.data = gridded_data
        gridded_dataset.model_data = gridded_model

        return gridded_dataset
