from nos.data import (
    TLDataset,
    TLDatasetCompact,
)


class TestTLDataset:
    def test_can_initialize(self, tl_csv_file):
        dataset = TLDataset(tl_csv_file)

        assert isinstance(dataset, TLDataset)

    def test_has_populated_tensors(self, tl_csv_file):
        dataset = TLDataset(tl_csv_file)

        assert dataset.x.nelement() > 0
        assert dataset.u.nelement() > 0
        assert dataset.y.nelement() > 0
        assert dataset.v.nelement() > 0


class TestTLDatasetCompact:
    def test_can_initialize(self, tl_csv_file):
        dataset = TLDatasetCompact(tl_csv_file)

        assert isinstance(dataset, TLDatasetCompact)

    def test_has_populated_tensors(self, tl_csv_file):
        dataset = TLDatasetCompact(tl_csv_file)

        assert dataset.x.nelement() > 0
        assert dataset.u.nelement() > 0
        assert dataset.y.nelement() > 0
        assert dataset.v.nelement() > 0
