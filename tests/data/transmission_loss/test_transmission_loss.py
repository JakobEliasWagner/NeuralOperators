import pytest

from nos.data import (
    TLDataset,
    TLDatasetCompact,
    TLDatasetCompactExp,
    TLDatasetCompactWave,
)


@pytest.mark.slow
@pytest.mark.parametrize("dataset_type", [TLDataset, TLDatasetCompact, TLDatasetCompactExp, TLDatasetCompactWave])
class TestTLCommon:
    def test_can_initialize(self, tl_paths, tl_dataset_sizes, dataset_type):
        for path in tl_paths:
            for size in tl_dataset_sizes:
                dataset = dataset_type(path=path, n_samples=size)
                assert isinstance(dataset, dataset_type)

    def test_shape_correct(self, tl_paths, tl_dataset_sizes, dataset_type):
        for path in tl_paths:
            for size in tl_dataset_sizes:
                dataset = dataset_type(path=path, n_samples=size)
                assert dataset.x.size(0) == dataset.y.size(0) == dataset.u.size(0) == dataset.v.size(0)
                assert dataset.x.size(1) == dataset.u.size(1)
                assert dataset.y.size(0) == dataset.v.size(0)
