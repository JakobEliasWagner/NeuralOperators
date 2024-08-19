import torch

from nos.transforms import (
    Log10Scale,
)


class TestLog10Scale:
    def test_can_initialize(self):
        transform = Log10Scale()
        assert isinstance(transform, Log10Scale)

    def test_forward_correct(self):
        x = torch.rand(10) + 1e-4
        transform = Log10Scale()

        out = transform(x)
        out_correct = torch.log10(x)

        assert torch.allclose(out, out_correct)

    def test_undo_correct(self):
        x = torch.rand(10) - 0.5

        transform = Log10Scale()

        out = transform.undo(x)
        out_correct = 10**x

        assert torch.allclose(out, out_correct)
