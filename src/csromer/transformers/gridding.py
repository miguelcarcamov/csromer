from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from scipy.special import i0

from ..utils.array_utils import asnumpy

if TYPE_CHECKING:
    from ..base import Dataset

EPSILON = 1e-12


def _kaiser_weight(u: float, half_width: float, beta: float) -> float:
    """Kaiser kernel weight for offset u (in grid units); 0 for |u| > half_width."""
    if abs(u) > half_width:
        return 0.0
    if half_width <= 0:
        return 1.0 if u == 0 else 0.0
    x = np.sqrt(1.0 - (u / half_width)**2)
    return float(i0(beta * x) / i0(beta))


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


@dataclass
class Gridding:
    """
    Grid non-uniform lambda² data onto a uniform lambda² grid.

    When d_lambda2 is provided (e.g. from Nyquist: π/(n_phi * d_phi)), the grid
    uses that spacing. The grid starts at dataset.l2_min (λ² > 0), not 0.
    If n is also provided, the grid has exactly n points
    (l2_min, l2_min+d_l2, ..., l2_min+(n-1)*d_l2). If n is not provided, n is
    computed from the dataset (phi_max, delta_phi, oversampling) to match
    max_faraday_depth.

    kernel: "box" = nearest-neighbor; "kaiser" = Kaiser window for smoother
    gridding and better peak preservation in Faraday depth space.
    """

    dataset: Optional["Dataset"] = None
    d_lambda2: Optional[float] = None
    n: Optional[int] = None
    kernel: Literal["box", "kaiser"] = "kaiser"
    gridding_kernel_half_width: float = 4.0
    gridding_kernel_beta: float = 2.5
    oversampling: float = 8.0

    def __post_init__(self) -> None:
        self.kernel = (self.kernel or "kaiser").strip().lower()
        if self.kernel not in ("box", "kaiser"):
            self.kernel = "kaiser"

    def run(self) -> "Dataset":
        if self.dataset is None:
            raise ValueError("Gridding requires dataset")
        gridded_dataset = copy.deepcopy(self.dataset)
        gridded_dataset.gridded = True

        l2_min = self.dataset.l2_min
        l2_max = self.dataset.l2_max
        if l2_min is None or l2_max is None:
            raise ValueError("Dataset must have lambda2 set for gridding")

        step = (float(self.d_lambda2) if self.d_lambda2 is not None else self.dataset.delta_l2_mean)
        if step is None or step <= 0:
            raise ValueError("Gridding requires d_lambda2 or dataset.delta_l2_mean")

        n = self.n
        if n is None:
            # Compute n from dataset (same logic as Parameter.calculate_cellsize)
            delta_phi_fwhm = self.dataset.delta_phi
            if delta_phi_fwhm is None:
                delta_phi_fwhm = (
                    2.0 * np.sqrt(3.0) / (l2_max - l2_min) if l2_max > l2_min else 2.0 /
                    (l2_max + l2_min)
                )
            phi_max = np.sqrt(3) / float(self.dataset.delta_l2_mean or 1e-20)
            phi_max = max(phi_max, float(delta_phi_fwhm) * 10.0)
            phi_r = float(delta_phi_fwhm) / self.oversampling
            temp = np.floor(2.0 * phi_max / phi_r)
            n = int(temp - np.mod(temp, 32))
            n = max(n, 32)

        # Grid starts at l2_min (λ² > 0), same step; Nyquist d_l2 unchanged
        l2_grid = l2_min + np.arange(n, dtype=np.float64) * step
        m_grid = len(l2_grid)

        l2_chan = np.asarray(asnumpy(self.dataset.lambda2))
        w_chan = np.asarray(self.dataset.w, dtype=np.float32)
        data_chan = np.asarray(self.dataset.data, dtype=np.complex64)
        model_chan = np.asarray(self.dataset.model_data, dtype=np.complex64)
        n_chan = len(l2_chan)

        gridded_data = np.zeros(m_grid, dtype=np.complex64)
        gridded_model = np.zeros(m_grid, dtype=np.complex64)
        gridded_w = np.zeros(m_grid, dtype=np.float32)

        if self.kernel == "box":
            # Bin index: (l2_chan - l2_min) / step
            l2_grid_pos = np.floor((l2_chan - l2_min) / step).astype(int)
            l2_grid_pos = np.clip(l2_grid_pos, 0, m_grid - 1)
            bincount_data = complex_bincount(l2_grid_pos, w_chan * data_chan)
            bincount_model = complex_bincount(l2_grid_pos, w_chan * model_chan)
            bincount_weights = np.bincount(l2_grid_pos, w_chan, minlength=m_grid)
            if len(bincount_data) < m_grid:
                bincount_data = np.pad(bincount_data, (0, m_grid - len(bincount_data)))
                bincount_model = np.pad(bincount_model, (0, m_grid - len(bincount_model)))
            unique_idx = np.unique(l2_grid_pos)
            gridded_data[unique_idx] = bincount_data[unique_idx]
            gridded_model[unique_idx] = bincount_model[unique_idx]
            gridded_w[unique_idx] = bincount_weights[unique_idx]
        else:
            half_w = float(self.gridding_kernel_half_width)
            beta = float(self.gridding_kernel_beta)
            for i in range(n_chan):
                l2_i = l2_chan[i]
                j_center = (l2_i - l2_min) / step
                j_lo = max(0, int(np.ceil(j_center - half_w)))
                j_hi = min(m_grid - 1, int(np.floor(j_center + half_w)))
                for j in range(j_lo, j_hi + 1):
                    u = j - j_center
                    kw = _kaiser_weight(u, half_w, beta)
                    if kw <= 0:
                        continue
                    gridded_data[j] += kw * w_chan[i] * data_chan[i]
                    gridded_model[j] += kw * w_chan[i] * model_chan[i]
                    gridded_w[j] += kw * w_chan[i]

        valid_idx = np.where(gridded_w > 0.0)
        gridded_data[valid_idx] /= gridded_w[valid_idx]
        gridded_model[valid_idx] /= gridded_w[valid_idx]

        gridded_dataset.lambda2 = l2_grid
        gridded_dataset.w = gridded_w.astype(np.float32)
        gridded_dataset.data = gridded_data.astype(np.complex64)
        gridded_dataset.model_data = gridded_model.astype(np.complex64)

        return gridded_dataset
