import torch
import pytest
from nos.physics import Laplace


@pytest.fixture(scope='function')
def derivative_pair():
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 100),
        torch.linspace(-1, 1, 100),
    )
    x = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    x = x.unsqueeze(0)  # 1 observation
    x.requires_grad = True
    u = torch.sin(x[:, :, 0]).unsqueeze(-1)
    return x, u


class TestLaplace:
    def test_can_initialize(self):
        _ = Laplace()

        assert True

    def test_can_forward(self, derivative_pair):
        x, u = derivative_pair

        laplace = Laplace()
        lpl_u = laplace(x, u)

        assert isinstance(lpl_u, torch.Tensor)

    def test_forward_shape_correct(self, derivative_pair):
        x, u = derivative_pair

        laplace = Laplace()
        lpl_u = laplace(x, u)

        assert u.shape == lpl_u.shape

    def test_forward_correct(self, derivative_pair):
        x, u = derivative_pair

        laplace = Laplace()
        lpl_u = laplace(x, u)

        assert torch.allclose(lpl_u, -u)
