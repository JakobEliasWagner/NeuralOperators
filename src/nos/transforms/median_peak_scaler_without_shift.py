import torch
import torch.nn as nn
from continuity.transforms import (
    Transform,
)


class MedianPeak(Transform):
    def __init__(
        self,
        src: torch.Tensor,
    ):
        super().__init__()
        peaks, _ = torch.max(torch.abs(src), dim=1)
        medians, _ = torch.median(peaks, dim=0)
        self.medians = nn.Parameter(medians.view(1, -1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / self.medians

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.medians
