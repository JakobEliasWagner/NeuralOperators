import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        act: nn.Module,
        dropout_p: float = 0.0,
        is_last: bool = False,
    ):
        super().__init__()

        self.net = nn.Sequential()
        depth = depth if not is_last else depth - 1
        for i in range(depth):
            self.net.add_module(f"linear_{i}", torch.nn.Linear(width, width))
            self.net.add_module(f"Act_{i}", act)

        if is_last:
            self.net.add_module(f"linear_{depth - 1}", torch.nn.Linear(width, width))

        if dropout_p > 0:
            self.net.add_module("Dropout", nn.Dropout(dropout_p))

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out + x


class ResNet(nn.Module):
    def __init__(self, width: int, depth: int, act: nn.Module, stride: int = 1, dropout_p: float = 0.0):
        super().__init__()

        assert depth % stride == 0
        n_blocks = depth // stride

        self.net = nn.Sequential()
        for i in range(n_blocks - 1):
            self.net.add_module(
                f"ResBlock_{i}",
                ResBlock(width=width, depth=stride, act=act, dropout_p=dropout_p),
            )
        self.net.add_module(
            f"ResBlock_{n_blocks - 1}",
            ResBlock(width=width, depth=stride, act=act, dropout_p=0.0, is_last=True),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
