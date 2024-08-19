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
        peaks, _ = torch.max(torch.abs(src), dim=1, keepdim=True)
        medians, _ = torch.median(peaks, dim=0, keepdim=True)
        self.medians = nn.Parameter(medians)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / self.medians

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.medians
