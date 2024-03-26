from nos.plots import (
    ModelData,
    MultiRunData,
    RunData,
)


class TestMultirunData:
    def test_can_initialize(self, exemplar_multirun_path):
        data = MultiRunData.from_dir(exemplar_multirun_path)
        assert isinstance(data, MultiRunData)


class TestRunData:
    def test_can_initialize(self, exemplar_run_path):
        data = RunData.from_dir(exemplar_run_path)
        assert isinstance(data, RunData)


class TestModelData:
    def test_can_initialize(self, exemplar_model_path):
        data = ModelData.from_dir(exemplar_model_path)
        assert isinstance(data, ModelData)
