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
    """
    Grid non-uniform lambda² data onto a uniform lambda² grid.

    When d_lambda2 is provided (e.g. from Nyquist: π/(n_phi * d_phi)), the grid
    uses that spacing. If n is also provided, the grid has exactly n points so
    that GriddedFFT1D can use the same phi grid (same length and resolution) as
    other methods. Otherwise uses dataset.delta_l2_mean.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        d_lambda2: float = None,
        n: int = None,
    ):
        """
        Args:
            dataset: Dataset with non-uniform (or uniform) lambda².
            d_lambda2: Optional uniform lambda² step (m²). If given, used for the
                gridded axis so that the result is compatible with a phi grid
                satisfying Nyquist d_phi * d_lambda2 = π/N. If None, uses
                dataset.delta_l2_mean.
            n: Optional number of grid points. When given with d_lambda2, the
                gridded grid has exactly n points (0, d_lambda2, ..., (n-1)*d_lambda2)
                so that the same param (same length and resolution) can be used
                for GriddedFFT1D as for DirectFourier1D/NUFFT1D.
        """
        self.dataset = dataset
        self.d_lambda2 = d_lambda2
        self.n = n

    def run(self):
        gridded_dataset = copy.deepcopy(self.dataset)
        gridded_dataset.gridded = True
        step = (
            float(self.d_lambda2)
            if self.d_lambda2 is not None
            else self.dataset.delta_l2_mean
        )
        if self.n is not None:
            # Fixed length grid: 0, d_l2, ..., (n-1)*d_l2 for GriddedFFT1D (no l2_ref in transform).
            # Dataset handles lambda²=0 defensively (nu/weights set without divide-by-zero).
            l2_grid = np.arange(0.0, self.n * step, step, dtype=np.float64)[: self.n]
        else:
            l2_grid = np.arange(
                start=0.0 + EPSILON,
                stop=np.max(self.dataset.lambda2),
                step=step,
            )
        l2_grid_pos = np.floor(self.dataset.lambda2 / step).astype(int)
        if self.n is not None:
            l2_grid_pos = np.clip(l2_grid_pos, 0, self.n - 1)
        m_grid = len(l2_grid)
        bincount_data = complex_bincount(l2_grid_pos, self.dataset.w * self.dataset.data)
        # model_data may be None or have a different length (e.g. after channel
        # removal where weights/data were shortened but model_data kept original
        # length). In that case, treat model_data as zero to avoid shape errors.
        if (
            self.dataset.model_data is not None
            and self.dataset.model_data.shape == self.dataset.data.shape
        ):
            bincount_model = complex_bincount(
                l2_grid_pos, self.dataset.w * self.dataset.model_data
            )
        else:
            bincount_model = np.zeros(m_grid, dtype=np.complex64)
        bincount_weights = np.bincount(l2_grid_pos, self.dataset.w, minlength=m_grid)
        # For complex bincount we need to pad to m_grid manually
        if len(bincount_data) < m_grid:
            bincount_data = np.pad(bincount_data, (0, m_grid - len(bincount_data)))
            bincount_model = np.pad(bincount_model, (0, m_grid - len(bincount_model)))
        unique_idx = np.unique(l2_grid_pos)
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
