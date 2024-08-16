import torch
from continuity.transforms import (
    Transform,
)


class Log10Scale(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = torch.log10(torch.abs(tensor))
        return out * torch.sign(tensor)

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return 10**tensor


class SymmetricLog10Scale(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = torch.log10(torch.abs(tensor) + 1)
        return out * torch.sign(tensor)

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        return 10**tensor
