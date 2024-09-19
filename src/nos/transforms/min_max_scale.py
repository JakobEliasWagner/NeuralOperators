import torch  # noqa: D100
from continuiti.transforms import (
    Transform,
)
from torch import nn


class MinMaxScale(Transform):
    """Min-max scale transformation."""

    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor) -> None:
        """Initialize.

        Args:
            min_value (torch.Tensor): Minimum values of the source distribution.
            max_value (torch.Tensor): Maximum values of the source distribution.

        """
        super().__init__()
        self.ndim = min_value.ndim

        self.min_value = nn.Parameter(min_value)
        self.max_value = nn.Parameter(max_value)

        delta = max_value - min_value
        delta[delta == 0] = 1.0
        self.delta = nn.Parameter(delta)

        target_min = -1.0 * torch.ones(min_value.shape)
        self.target_min = nn.Parameter(target_min)

        target_max = 1.0 * torch.ones(min_value.shape)

        self.target_delta = nn.Parameter(target_max - target_min)

    def _is_batched(self, tensor: torch.Tensor) -> bool:
        return tensor.ndim == (self.ndim + 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform tensor."""
        if self._is_batched(tensor):
            # observation dimension 0
            v_m = self.min_value.unsqueeze(0)
            d = self.delta.unsqueeze(0)
            tgt_d = self.target_delta.unsqueeze(0)
            tgt_m = self.target_min.unsqueeze(0)
        else:
            # single observation
            v_m = self.min_value
            d = self.delta
            tgt_d = self.target_delta
            tgt_m = self.target_min

        return ((tensor - v_m) / d) * tgt_d + tgt_m

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo transformation."""
        if self._is_batched(tensor):
            tgt_m = self.target_min.unsqueeze(0)
            tgt_d = self.target_delta.unsqueeze(0)
            d = self.delta.unsqueeze(0)
            v_m = self.min_value.unsqueeze(0)
        else:
            tgt_m = self.target_min
            tgt_d = self.target_delta
            d = self.delta
            v_m = self.min_value

        return (tensor - tgt_m) / tgt_d * d + v_m
