import torch

from nos.transforms import (
    MedianPeak,
)


class TestMedianPeakScaler:
    x = torch.rand(7, 11, 13)
    y = torch.rand(17, 19, 13)

    def test_can_initialize(self):
        transform = MedianPeak(self.x)

        assert isinstance(transform, MedianPeak)

    def test_can_forward(self):
        transform = MedianPeak(self.x)

        out = transform(self.y)

        assert isinstance(out, torch.Tensor)

    def test_can_undo(self):
        transform = MedianPeak(self.x)

        out = transform.undo(self.y)

        assert isinstance(out, torch.Tensor)
