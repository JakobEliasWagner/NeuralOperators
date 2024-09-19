from nos.data import ConstBoundaryDataset, InverseConstBoundaryDataset


class TestPulsatingSphereDataset:
    def test_can_initialize(self, pressure_file):
        dataset = ConstBoundaryDataset(
            dataset_path=pressure_file,
            observations=42,
        )

        assert isinstance(dataset, ConstBoundaryDataset)

    def test_has_populated_tensors(self, pressure_file):
        n_observations = 42
        dataset = ConstBoundaryDataset(
            dataset_path=pressure_file,
            observations=n_observations,
        )

        for tsr in [dataset.x, dataset.y, dataset.u, dataset.v]:
            assert tsr.size(0) == n_observations

        assert dataset.x.size(-1) == dataset.u.size(-1)
        assert dataset.y.size(-1) == dataset.v.size(-1)


class TestInversePulsatingSphereDataset:
    def test_can_initialize(self, pressure_file):
        dataset = InverseConstBoundaryDataset(
            dataset_path=pressure_file,
            observations=42,
        )

        assert isinstance(dataset, InverseConstBoundaryDataset)

    def test_has_populated_tensors(self, pressure_file):
        n_observations = 42
        dataset = ConstBoundaryDataset(
            dataset_path=pressure_file,
            observations=n_observations,
        )

        for tsr in [dataset.x, dataset.y, dataset.u, dataset.v]:
            assert tsr.size(0) == n_observations

        assert dataset.x.size(-1) == dataset.u.size(-1)
        assert dataset.y.size(-1) == dataset.v.size(-1)
