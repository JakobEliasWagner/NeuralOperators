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
        ks = k.squeeze() ** 2
        ks = ks.reshape(-1, 1, 1)
        ks = ks.expand(v.size(0), 1, 1)
        lpl = self.laplace(y, v)
        return lpl + ks * v


class HelmholtzDomainMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pde = HelmholtzDomainResidual()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        residual = self.pde(y, v, k)
        residual = residual**2
        return torch.mean(residual)
