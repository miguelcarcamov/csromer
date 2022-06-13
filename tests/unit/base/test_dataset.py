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
