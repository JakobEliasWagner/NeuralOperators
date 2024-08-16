import torch
import torch.nn as nn
from continuity.transforms import (
    Transform,
)


class MinMaxScale(Transform):
    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor):
        super().__init__()

        self.min_value = nn.Parameter(min_value)
        self.max_value = nn.Parameter(max_value)

        delta = max_value - min_value
        delta[delta == 0] = 1.0
        self.delta = nn.Parameter(delta)

        target_min = -1.0 * torch.ones(1)
        self.target_min = nn.Parameter(target_min)
        target_max = 1.0 * torch.ones(1)
        self.target_max = nn.Parameter(target_max)
        self.target_delta = nn.Parameter(target_max - target_min)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor - self.min_value) / self.delta) * self.target_delta + self.target_min

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.target_min) / self.target_delta * self.delta + self.min_value
