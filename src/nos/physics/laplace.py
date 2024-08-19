import torch
from continuiti.operators import (
    Operator,
)
from continuiti.pde import (
    Grad,
)


class Laplace(Operator):
    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        second_derivatives = []
        derivative = Grad()(x, u)
        for dim in range(x.size(-1)):
            second_derivatives.append(Grad()(x, derivative[:, :, dim])[:, :, dim])
        second_derivatives = torch.stack(second_derivatives, dim=-1)
        return torch.sum(second_derivatives, dim=-1, keepdim=True)
