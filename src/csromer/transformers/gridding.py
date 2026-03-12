from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import i0

if TYPE_CHECKING:
    from ..base import Dataset

EPSILON = 1e-12


def _kaiser_weight(u: float, half_width: float, beta: float) -> float:
    """Kaiser kernel weight for offset u (in grid units); 0 for |u| > half_width."""
    if abs(u) > half_width:
        return 0.0
    if half_width <= 0:
        return 1.0 if u == 0 else 0.0
    x = np.sqrt(1.0 - (u / half_width) ** 2)
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


class Gridding:
    """
    Grid non-uniform lambda² data onto a uniform lambda² grid.
    
    When d_lambda2 is provided (e.g. from Nyquist: π/(n_phi * d_phi)), the grid
    uses that spacing. If n is also provided, the grid has exactly n points so
    that GriddedFFT1D can use the same phi grid (same length and resolution) as
    other methods. Otherwise uses dataset.delta_l2_mean.
    
    kernel: "box" = nearest-neighbor (current behavior); "kaiser" = Kaiser window
    for smoother gridding and better peak preservation in Faraday depth space.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        d_lambda2: float = None,
        n: int = None,
        kernel: Literal["box", "kaiser"] = "kaiser",
        gridding_kernel_half_width: float = 4.0,
        gridding_kernel_beta: float = 2.5,
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
            kernel: "box" (nearest-neighbor) or "kaiser" (Kaiser window).
            gridding_kernel_half_width: Half-width in grid units for Kaiser (ignored for box).
            gridding_kernel_beta: Kaiser shape parameter (ignored for box).
        """
        self.dataset = dataset
        self.d_lambda2 = d_lambda2
        self.n = n
        self.kernel = (kernel or "kaiser").strip().lower()
        if self.kernel not in ("box", "kaiser"):
            self.kernel = "kaiser"
        self.gridding_kernel_half_width = gridding_kernel_half_width
        self.gridding_kernel_beta = gridding_kernel_beta

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
        m_grid = len(l2_grid)
        l2_chan = np.asarray(self.dataset.lambda2)
        w_chan = np.asarray(self.dataset.w, dtype=np.float64)
        data_chan = np.asarray(self.dataset.data, dtype=np.complex128)
        model_chan = np.asarray(self.dataset.model_data, dtype=np.complex128)
        n_chan = len(l2_chan)

        gridded_data = np.zeros(m_grid, dtype=np.complex128)
        gridded_model = np.zeros(m_grid, dtype=np.complex128)
        gridded_w = np.zeros(m_grid, dtype=np.float64)

        if self.kernel == "box":
            l2_grid_pos = np.floor(l2_chan / step).astype(int)
            if self.n is not None:
                l2_grid_pos = np.clip(l2_grid_pos, 0, self.n - 1)
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
            # Kaiser (or other window): spread each channel to nearby grid points
            half_w = float(self.gridding_kernel_half_width)
            beta = float(self.gridding_kernel_beta)
            for i in range(n_chan):
                l2_i = l2_chan[i]
                j_center = l2_i / step
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

        gridded_dataset.lambda2 = l2_grid  # updates m to m_grid
        gridded_dataset.w = gridded_w.astype(np.float32)
        gridded_dataset.data = gridded_data.astype(np.complex64)
        gridded_dataset.model_data = gridded_model.astype(np.complex64)
        # effective_n (Kish: (sum w)^2 / sum(w^2)) is a property computed from w, so it
        # automatically reflects the gridded effective sample count when the objective uses it.

        return gridded_dataset
