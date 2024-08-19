import torch

from nos.transforms import CenterQuantileScaler


class TestCenterQuantileScaler:
    x = torch.rand(7, 89, 13)
    y = torch.rand(17, 97, 13)

    def test_can_init(self):
        transform = CenterQuantileScaler(self.x)

        assert isinstance(transform, CenterQuantileScaler)

    def test_can_forward(self):
        transform = CenterQuantileScaler(self.x)

        out = transform(self.y)

        assert isinstance(out, torch.Tensor)

    def test_forward_centered(self):
        transform = CenterQuantileScaler(self.x)

        out = transform(self.y)

        assert torch.isclose(torch.median(out), torch.zeros(1), atol=1e-1)

    def test_can_undo(self):
        transform = CenterQuantileScaler(self.x)

        out = transform.undo(self.y)

        assert isinstance(out, torch.Tensor)
