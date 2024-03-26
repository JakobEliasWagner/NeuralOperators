from nos.data.indicator_function import (
    IndicatorTLDataset,
)


class TestIndicatorToTl:
    def test_can_initialize(self, tl_csv_file):
        dataset = IndicatorTLDataset(path=tl_csv_file)
        assert isinstance(dataset, IndicatorTLDataset)

    def test_shapes_correct(self, tl_csv_file):
        dataset = IndicatorTLDataset(path=tl_csv_file, n_box_samples=509)
        assert dataset.shapes.x.num == 509
        assert dataset.shapes.x.dim == 2
        assert dataset.shapes.u.num == 509
        assert dataset.shapes.u.dim == 1
