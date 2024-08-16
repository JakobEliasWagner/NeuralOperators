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
        lpl = []
        for dim in range(v.size(-1)):
            lpl.append(self.laplace(y, v[:, :, dim]))
        lpl = torch.stack(lpl, dim=-1)
        return lpl + ks * v


class HelmholtzDomainMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pde = HelmholtzDomainResidual()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        residual = self.pde(y, v, k)
        residual = residual**2
        return torch.mean(residual)


class HelmholtzDomainMedianSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pde = HelmholtzDomainResidual()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        residual = self.pde(y, v, k)
        residual = residual**2
        return torch.median(residual)
