import torch
from continuity.operators import (
    Operator,
)
from continuity.pde import (
    Grad,
)


class Laplace(Operator):
    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        gradients = Grad()(x, u)
        return torch.sum(Grad()(x, gradients), dim=-1, keepdim=True)
