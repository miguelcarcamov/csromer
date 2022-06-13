from csromer.base import Dataset


class TestDataset:
    dataset = Dataset()

    def test_empty_dataset(self):
        assert self.dataset is not None

    def test_empty_dataset_k(self):
        assert self.dataset.k is None
