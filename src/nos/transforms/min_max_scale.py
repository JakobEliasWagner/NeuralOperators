import torch
from continuity.transforms import (
    Transform,
)


class MinMaxScale(Transform):
    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor):
        self.min_value = min_value
        self.max_value = max_value
        self.delta = max_value - min_value

        self.target_min = -1.0
        self.target_max = 1.0
        self.target_delta = self.target_max - self.target_min

        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor - self.min_value) / self.delta) * self.target_delta + self.target_min

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.target_min) / self.target_delta * self.delta + self.min_value
