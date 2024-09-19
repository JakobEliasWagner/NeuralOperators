import torch  # noqa: D100
from torch import nn


class WeightSchedulerLinear(nn.Module):
    """Scheduler that linearly increases the fraction of the pde term in the loss within an interval."""

    def __init__(self, pde_loss_start: int, pde_loss_full: int) -> None:
        """Initialize.

        Args:
            pde_loss_start (int): Epoch to indicate the start of the linear increase.
            pde_loss_full (int): Epoch in which the fraction is 100%.

        """
        super().__init__()
        self.pde_loss_start = pde_loss_start
        self.pde_loss_full = pde_loss_full
        self.pde_delta = pde_loss_full - pde_loss_start

    def _get_data_weight(self, epoch: torch.Tensor) -> torch.Tensor:
        """Get data weight contribution."""
        return torch.ones(epoch.shape)

    def _get_pde_weight(self, epoch: torch.Tensor) -> torch.Tensor:
        """Get pde weight contribution."""
        out = (epoch - self.pde_loss_start) / self.pde_delta
        out[out < 0.0] = 0.0
        out[out > 1] = 1.0
        return out * 1e-3

    def forward(self, epoch: float | torch.Tensor) -> torch.Tensor:
        """Evaluate the weights for the current epoch.

        Args:
            epoch (float | torch.tensor): Epoch or epochs.

        Returns:
            torch.tensor: weights for the epoch or epochs.

        """
        if not isinstance(epoch, torch.Tensor):
            epoch = epoch * torch.ones(1)

        data_weights = self._get_data_weight(epoch)
        pde_weights = self._get_pde_weight(epoch)

        return torch.stack([data_weights, pde_weights], dim=1)
