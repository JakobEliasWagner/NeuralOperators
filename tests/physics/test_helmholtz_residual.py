import pytest
import torch

from nos.physics import (
    HelmholtzDomainMSE,
    HelmholtzDomainResidual,
)


@pytest.fixture
def wave_1d():
    n_obs = 13
    n_eval = 31

    x = torch.rand(n_obs, n_eval, 1)
    x.requires_grad = True

    ks = 10 * torch.rand(n_obs).reshape(-1, 1, 1)

    u = torch.sin(ks * x)

    return x, u, ks


@pytest.fixture
def wave_1d_wrong():
    n_obs = 13
    n_eval = 31

    x = torch.rand(n_obs, n_eval, 1)
    x.requires_grad = True

    ks = 10 * torch.rand(n_obs).reshape(-1, 1, 1)

    u = torch.sin(ks * 1.1 * x)

    return x, u, ks


class TestHelmholtzDomainResidual:
    def test_can_initialize(self):
        res = HelmholtzDomainResidual()

        assert isinstance(res, HelmholtzDomainResidual)

    def test_can_forward(self, wave_1d):
        x, u, ks = wave_1d

        res = HelmholtzDomainResidual()
        res_val = res(x, u, ks)

        assert isinstance(res_val, torch.Tensor)

    def test_forward_correct(self, wave_1d):
        x, u, ks = wave_1d

        res = HelmholtzDomainResidual()
        res_val = res(x, u, ks)

        assert torch.allclose(res_val, torch.zeros(res_val.shape), atol=1e-5)

    def test_forward_wrong(self, wave_1d_wrong):
        x, u, ks = wave_1d_wrong

        res = HelmholtzDomainResidual()
        res_val = res(x, u, ks)

        assert not torch.any(torch.isclose(res_val, torch.zeros(res_val.shape), atol=1e-5))


class TestHelmholtzDomainMSE:
    def test_can_initialize(self):
        res = HelmholtzDomainMSE()

        assert isinstance(res, HelmholtzDomainMSE)

    def test_can_forward(self, wave_1d):
        x, u, ks = wave_1d

        res = HelmholtzDomainMSE()
        res_val = res(x, u, ks)

        assert isinstance(res_val, torch.Tensor)

    def test_forward_correct(self, wave_1d):
        x, u, ks = wave_1d

        res = HelmholtzDomainMSE()
        res_val = res(x, u, ks)

        acceptable = 1e-8
        assert res_val < acceptable

    def test_forward_wrong(self, wave_1d_wrong):
        x, u, ks = wave_1d_wrong

        res = HelmholtzDomainMSE()
        res_val = res(x, u, ks)

        assert res_val > 1.0
