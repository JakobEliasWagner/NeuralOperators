import torch
from continuity.discrete import (
    UniformBoxSampler,
)

from nos.data.indicator_function.params_to_indicator import (
    params_to_indicator,
)


class TestParamsToIndicator:
    def test_shapes_correct(self):
        params = torch.rand(13, 1, 3)
        params, _ = torch.sort(params, descending=True)
        x, indicator = params_to_indicator(params, UniformBoxSampler([0.0, 0.0], [1.0, 1.0]), 509)

        assert x.size(0) == params.size(0) == indicator.size(0) == 13
        assert x.size(1) == indicator.size(1) == 509
        assert indicator.size(2) == 1
