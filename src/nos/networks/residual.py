import torch  # noqa: D100
from torch import nn


class ResBlock(nn.Module):
    """Single Res-Net block."""

    def __init__(
        self,
        width: int,
        depth: int,
        act: nn.Module,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize.

        Args:
            width (int): Width of the res-net-block.
            depth (int): depth of the res-net block.
            act (nn.Module): Activation function applied after each layer.
            dropout_p (float, optional): Temperature parameter controling the dropout applied after the entire block.
                Defaults to 0.0.

        """
        super().__init__()

        self.net = nn.Sequential()
        for i in range(depth):
            self.net.add_module(f"linear_{i}", torch.nn.Linear(width, width))
            self.net.add_module(f"norm_{i}", torch.nn.LayerNorm(width))
            self.net.add_module(f"Act_{i}", act)

        if dropout_p > 0.0:
            self.net.add_module("Dropout", nn.Dropout(dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        out = self.net(x)
        return out + x


class ResNet(nn.Module):
    """Residual network."""

    def __init__(self, width: int, depth: int, act: nn.Module, stride: int = 1, dropout_p: float = 0.0) -> None:
        """Initialize.

        Args:
            width (int): Width of all layers.
            depth (int): Depth of the entire ResNet.
            act (nn.Module): Activation function.
            stride (int, optional): Stride of the residual connections. Defaults to 1.
            dropout_p (float, optional): Temperature parameter controlling dropout. Defaults to 0.0.

        """
        super().__init__()

        n_blocks = depth // stride

        self.net = nn.Sequential()
        for i in range(n_blocks - 1):
            self.net.add_module(
                f"ResBlock_{i}",
                ResBlock(width=width, depth=stride, act=act, dropout_p=dropout_p),
            )
        # last block does should not have an activation function
        if stride > 1:
            self.net.add_module(
                f"ResBlock_{n_blocks - 1}",
                ResBlock(width=width, depth=stride - 1, act=act, dropout_p=0.0),
            )
        self.net.add_module(
            f"Linear_{depth - 1}",
            torch.nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass throught the ResNet."""
        return self.net(x)
