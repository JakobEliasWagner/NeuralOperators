import torch  # noqa: D100
from continuiti.operators import Operator
from continuiti.pde import Grad


class Laplace(Operator):
    """Laplace operator."""

    def forward(self, x: torch.Tensor, u: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        """Evaluate the laplace operator.

        Args:
            x (torch.Tensor): Locations.
            u (torch.Tensor): Values. Requires gradient.

        Returns:
            torch.Tensor: Laplace operator evaluated in locations.

        """
        derivative = Grad()(x, u)
        second_derivatives = [Grad()(x, derivative[:, :, dim])[:, :, dim] for dim in range(x.size(-1))]
        sd_tensor = torch.stack(second_derivatives, dim=-1)
        return torch.sum(sd_tensor, dim=-1, keepdim=True)
