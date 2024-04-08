from typing import (
    Union,
)

import torch
import torch.nn as nn


class WeightSchedulerLinear(nn.Module):
    def __init__(self, pde_loss_start: int, pde_loss_full: int):
        super().__init__()
        self.pde_loss_start = pde_loss_start
        self.pde_loss_full = pde_loss_full
        self.pde_delta = pde_loss_full - pde_loss_start

    def _get_data_weight(self, epoch: torch.tensor) -> torch.tensor:
        return torch.ones(epoch.shape)

    def _get_pde_weight(self, epoch: torch.tensor) -> torch.tensor:
        out = (epoch - self.pde_loss_start) / self.pde_delta
        out[out < 0.0] = 0.0
        out[out > 1] = 1.0
        return out * 1e-3

    def forward(self, epoch: Union[float, torch.tensor]) -> torch.tensor:
        if isinstance(epoch, int):
            epoch = epoch * torch.ones(1)

        data_weights = self._get_data_weight(epoch)
        pde_weights = self._get_pde_weight(epoch)

        return torch.stack([data_weights, pde_weights], dim=1)
