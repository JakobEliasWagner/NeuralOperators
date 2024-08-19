import torch

from nos.transforms import QuantileScaler


class TestQuantileScaler:
    x = torch.rand(7, 11, 13)
    y = torch.rand(17, 19, 13)

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
        mod_x[0, 0, 0] = 1e+10
        transform_mod = QuantileScaler(mod_x, n_quantile_intervals=5)

        out = transform(self.y)
        mod_out = transform_mod(self.y)

        assert torch.isclose(torch.median(out), torch.median(mod_out))
