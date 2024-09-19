import torch  # noqa: D100
from torch import nn

from nos.physics.laplace import Laplace


class HelmholtzDomainResidual(nn.Module):
    """Helmholtz residual for the interior equation."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.laplace = Laplace()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual.

        Args:
            y (torch.Tensor): Evaluation locations.
            v (torch.Tensor): Evaluation values.
            k (torch.Tensor): Wave numbers.

        Returns:
            torch.Tensor: Residual with shape (num_observations, y_dim, 1).

        """
        ks = k.squeeze() ** 2
        ks = ks.reshape(-1, 1, 1)
        ks = ks.expand(v.size(0), 1, 1)
        lpl = self.laplace(y, v)
        return lpl + ks * v


class HelmholtzDomainMSE(nn.Module):
    """Helmholtz residual evaluated in mean square error metric."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.pde = HelmholtzDomainResidual()

    def forward(self, y: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Evaluate mean squared residual.

        Args:
            y (torch.Tensor): Evaluation locations.
            v (torch.Tensor): Evaluation values.
            k (torch.Tensor): Wave numbers.

        Returns:
            torch.Tensor: Scalar valued tensor containing the mean squared residual.

        """
        residual = self.pde(y, v, k)
        residual = residual**2
        return torch.mean(residual)
