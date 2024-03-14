import torch

from continuity.transforms import (
    Transform,
)


class MinMaxScale(Transform):
    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.min_value) / (self.max_value - self.min_value)
