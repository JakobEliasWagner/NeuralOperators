import torch

from nos.transforms import (
    QuantileScaler,
)


class TestQuantileScaler:
    x = torch.rand(7, 13, 11)
    y = torch.rand(17, 13, 19)

    def test_can_init(self):
        transform = QuantileScaler(self.x)

        assert isinstance(transform, QuantileScaler)

    def test_can_forward(self):
        transform = QuantileScaler(self.x)

        out = transform(self.y)

        assert isinstance(out, torch.Tensor)

    def test_can_undo(self):
        transform = QuantileScaler(self.x)

        out = transform.undo(self.y)

        assert isinstance(out, torch.Tensor)

    def test_stable_to_outliers(self):
        transform = QuantileScaler(self.x, n_quantile_intervals=5)

        mod_x = self.x
        mod_x[0, 0, 0] = 1e10
        transform_mod = QuantileScaler(mod_x, n_quantile_intervals=5)

        out = transform(self.y)
        mod_out = transform_mod(self.y)

        assert torch.isclose(torch.median(out), torch.median(mod_out), atol=1e-1)

    def test_batched_forward(self):
        src = torch.rand(3, 5, 7)
        tf = QuantileScaler(src)

        batched_sample = torch.rand(11, 5, 7)

        out = tf(batched_sample)

        assert out.shape == batched_sample.shape

    def test_observation_forward(self):
        src = torch.rand(3, 5, 7)
        tf = QuantileScaler(src)

        obs_sample = torch.rand(5, 7)

        out = tf(obs_sample)

        assert out.shape == obs_sample.shape

    def test_batched_undo(self):
        src = torch.rand(3, 5, 7)
        tf = QuantileScaler(src)

        batched_sample = torch.rand(11, 5, 7)

        out = tf.undo(batched_sample)

        assert out.shape == batched_sample.shape

    def test_observation_undo(self):
        src = torch.rand(3, 5, 7)
        tf = QuantileScaler(src)

        obs_sample = torch.rand(5, 7)

        out = tf.undo(obs_sample)

        assert out.shape == obs_sample.shape
