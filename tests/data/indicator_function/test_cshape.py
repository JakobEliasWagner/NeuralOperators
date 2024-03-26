import torch

from nos.data.indicator_function import (
    CShape,
)


class TestCShape:
    def test_can_initialize(self):
        shape = CShape()
        assert isinstance(shape, CShape)

    def test_distance_correct(self):
        shape = CShape(outer_radius=2 * torch.ones(1), inner_radius=torch.ones(1), grid_size=4 * torch.ones(1))
        x = 2 * torch.ones(1, 2)
        distance = shape(x)
        assert torch.allclose(distance, torch.ones(1))

        x = torch.tensor([[2.0, 1.0]])
        distance = shape(x)
        assert torch.allclose(distance, torch.zeros(1), atol=1e-2)

    def test_inside_negative(self):
        shape = CShape(outer_radius=2 * torch.ones(1), inner_radius=torch.ones(1), grid_size=4 * torch.ones(1))
        x = torch.rand(100, 2) / 2 + torch.tensor([1.0, 0.5])
        distance = shape(x)

        assert torch.all(torch.less_equal(distance, 0))
