from ..base.dataset import Dataset
import numpy as np
import copy


class Gridding:
    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset

    def run(self):
        gridded_dataset = copy.deepcopy(self.dataset)
        l2_grid = np.arange(start=0.0, stop=np.max(self.dataset.lambda2), step=self.dataset.delta_l2_mean)
        m_grid = len(l2_grid)
        gridded_data = np.zeros(m_grid, dtype=np.complex64)
        gridded_model = np.zeros(m_grid, dtype=np.complex64)
        gridded_w = np.zeros(m_grid, dtype=np.float32)

        for i in range(0, len(self.dataset.lambda2)):
            l2_grid_pos = int(np.floor(self.dataset.lambda2[i] / self.dataset.delta_l2_mean))
            if 0 <= l2_grid_pos < m_grid:
                gridded_data[l2_grid_pos] += self.dataset.w[i] * self.dataset.data[i]
                gridded_model[l2_grid_pos] += self.dataset.w[i] * self.dataset.model_data[i]
                gridded_w[l2_grid_pos] += self.dataset.w[i]

        valid_idx = np.where(gridded_w > 0.0)
        gridded_data[valid_idx] /= gridded_w[valid_idx]
        gridded_model[valid_idx] /= gridded_w[valid_idx]

        gridded_dataset.lambda2 = l2_grid
        gridded_dataset.w = gridded_w
        gridded_dataset.data = gridded_data
        gridded_dataset.model_data = gridded_model

        return gridded_dataset
