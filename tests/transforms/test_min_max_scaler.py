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

    def test_scale_in_bounds(self):
        vec = torch.rand(100)
        tf = MinMaxScale(torch.min(vec), torch.max(vec))
        out = tf(vec)

        assert torch.all(torch.greater_equal(out, -torch.ones(vec.shape)))
        assert torch.all(torch.less_equal(out, torch.ones(vec.shape)))
        assert torch.min(out) == -1
        assert torch.max(out) == 1

    def test_batched_forward(self):
        src = torch.rand(3, 5, 7)

        min_val, _ = torch.min(src, dim=0)
        max_val, _ = torch.max(src, dim=0)

        tf = MinMaxScale(min_val, max_val)

        batched_sample = torch.rand(11, 5, 7)

        out = tf(batched_sample)

        assert out.shape == batched_sample.shape

    def test_observation_forward(self):
        src = torch.rand(3, 5, 7)

        min_val, _ = torch.min(src, dim=0)
        max_val, _ = torch.max(src, dim=0)

        tf = MinMaxScale(min_val, max_val)

        obs_sample = torch.rand(5, 7)

        out = tf(obs_sample)

        assert out.shape == obs_sample.shape

    def test_batched_undo(self):
        src = torch.rand(3, 5, 7)

        min_val, _ = torch.min(src, dim=0)
        max_val, _ = torch.max(src, dim=0)

        tf = MinMaxScale(min_val, max_val)

        batched_sample = torch.rand(11, 5, 7)

        out = tf.undo(batched_sample)

        assert out.shape == batched_sample.shape

    def test_observation_undo(self):
        src = torch.rand(3, 5, 7)

        min_val, _ = torch.min(src, dim=0)
        max_val, _ = torch.max(src, dim=0)

        tf = MinMaxScale(min_val, max_val)

        obs_sample = torch.rand(5, 7)

        out = tf.undo(obs_sample)

        assert out.shape == obs_sample.shape
