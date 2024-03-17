from typing import (
    List,
)

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, width: int, depth: int, act: nn.Module):
        super().__init__()

        self.net = nn.Sequential()
        for i in range(depth):
            self.net.add_module(f"linear_{i}", torch.nn.Linear(width, width))
            self.net.add_module(f"Act_{i}", act)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out + x


class ResNet(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        act: nn.Module,
        stride: int = 1,
        transition_transformations: List[nn.Module] = None,
    ):
        super().__init__()

        if transition_transformations is None:
            transition_transformations = []

        assert depth % stride == 0
        n_blocks = depth // stride

        self.net = nn.Sequential()
        for i in range(n_blocks):
            self.net.add_module(f"ResBlock_{i}", ResBlock(width=width, depth=stride, act=act))
            for j, transformation in enumerate(transition_transformations):
                self.net.add_module(f"Transformation_{i}_{j}", transformation())

    def forward(self, x: torch.Tensor):
        return self.net(x)
