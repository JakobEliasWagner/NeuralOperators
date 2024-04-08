import torch
import torch.nn as nn

from .laplace import (
    Laplace,
)


class HelmholtzDomainResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k = k**2
        k = k.reshape(-1, 1, 1)
        k = k.expand(v.shape)
        return self.laplace(y, v) + k**2 * v


class HelmholtzDomainMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pde = HelmholtzDomainResidual()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        residual = self.pde(y, v, k)
        residual = residual**2
        return torch.mean(residual)
