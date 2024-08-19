import torch

from nos.transforms import (
    MinMaxScale,
)


class TestMinMaxScale:
    def test_can_initialize(self):
        scaler = MinMaxScale(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
        assert isinstance(scaler, MinMaxScale)

    def test_can_forward(self):
        scaler = MinMaxScale(torch.zeros(2, 2), torch.ones(2, 2))

        x = torch.rand(2, 2)

        out = scaler(x)

        assert isinstance(out, torch.Tensor)

    def test_forward_correct(self):
        scaler = MinMaxScale(-torch.ones(1, 2, 1), 2 * torch.ones(1, 2, 1))

        x = torch.linspace(-1, 2, 30)
        x = x.reshape(1, 2, -1)

        out = scaler(x)

        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)
        assert torch.sum(torch.isclose(out, -torch.ones(out.shape))) == 1
        assert torch.sum(torch.isclose(out, torch.ones(out.shape))) == 1

    def test_can_undo(self):
        scaler = MinMaxScale(-torch.ones(1, 2, 1), 2 * torch.ones(1, 2, 1))

        x = torch.linspace(-1, 2, 30)

        out = scaler.undo(x)

        assert isinstance(out, torch.Tensor)
