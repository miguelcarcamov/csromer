import numpy as np

from csromer.base import Dataset


class TestDataset:
    dataset = Dataset()

    def test_empty_dataset(self):
        assert self.dataset is not None

    def test_empty_dataset_k(self):
        assert self.dataset.k is None

    def test_empty_dataset_nu(self):
        assert self.dataset.nu is None

    def test_empty_dataset_lambda2(self):
        assert self.dataset.lambda2 is None


class TestDatasetWeights:
    """Tests for w_q, w_u, w_p (harmonic mean) weight handling."""

    def test_w_p_harmonic_mean(self):
        lam2 = np.linspace(0.01, 0.1, 5) ** 2
        d = Dataset(lambda2=lam2)
        d.w_q = np.array([1.0, 2.0, 2.0, 2.0, 1.0])
        d.w_u = np.array([2.0, 2.0, 1.0, 2.0, 2.0])
        expected = 2.0 / (1.0 / d.w_q + 1.0 / d.w_u)
        np.testing.assert_allclose(d.w, expected)
        np.testing.assert_allclose(d.w_p, expected)

    def test_w_only_backward_compat(self):
        lam2 = np.linspace(0.01, 0.1, 4) ** 2
        d = Dataset(lambda2=lam2)
        d.w = np.ones(4) * 0.5
        assert d.w is not None
        assert d.w_q is None
        assert d.w_u is None
