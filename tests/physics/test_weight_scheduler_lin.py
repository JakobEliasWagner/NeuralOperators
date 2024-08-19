import torch

from nos.physics import (
    WeightSchedulerLinear,
)


class TestWeightSchedulerLin:
    def test_can_initialize(self):
        scheduler = WeightSchedulerLinear(1, 2)

        assert isinstance(scheduler, WeightSchedulerLinear)

    def test_data_weight_correct(self):
        scheduler = WeightSchedulerLinear(1, 2)

        epochs = torch.arange(0, 100)
        weight = scheduler._get_data_weight(epochs)

        assert isinstance(weight, torch.Tensor)
        assert torch.allclose(weight, torch.ones(weight.shape))

    def test_pde_weight_correct(self):
        scheduler = WeightSchedulerLinear(10, 20)

        epochs = torch.arange(0, 100)
        weight = scheduler._get_pde_weight(epochs)

        assert torch.allclose(weight[0:10], torch.zeros(10))
        assert torch.allclose(weight[20:], 1e-3 * torch.ones(80))
        assert torch.allclose(weight[10:21], 1e-3 * (epochs[10:21] - 10.0) / 10.0)

    def test_forward_weight_correct(self):
        scheduler = WeightSchedulerLinear(10, 20)

        epochs = torch.arange(0, 100)
        weight = scheduler(epochs)

        assert torch.allclose(weight[0:10, 1], torch.zeros(10))
        assert torch.allclose(weight[20:, 1], 1e-3 * torch.ones(80))
        assert torch.allclose(weight[10:21, 1], 1e-3 * (epochs[10:21] - 10.0) / 10.0)

        assert torch.allclose(weight[:, 0], torch.ones(100))

    def test_forward_float_weight_correct(self):
        scheduler = WeightSchedulerLinear(10, 20)

        weight = scheduler(15.0)

        assert torch.allclose(weight, torch.tensor([1.0, 5e-4]))
