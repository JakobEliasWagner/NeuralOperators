from typing import (
    Union,
)

import torch
import torch.nn as nn
from continuiti.transforms import (
    Transform,
)


class CenterQuantileScaler(Transform):
    def __init__(
        self,
        src: torch.Tensor,
        center_interval: float = 0.5,
        target_mean: Union[float, torch.Tensor] = 0.0,
        target_std: Union[float, torch.Tensor] = 1.0,
    ):
        super().__init__()

        if isinstance(target_mean, float):
            target_mean = target_mean * torch.ones(1)
        if isinstance(target_std, float):
            target_std = target_std * torch.ones(1)
        self.target_mean = target_mean
        self.target_std = target_std

        assert 1.0 >= center_interval > 0

        self.n_dim = src.size(-1)

        q = torch.tensor([center_interval / 2, 0.5, 1 - center_interval / 2])

        # source "distribution"
        quantile_points = torch.quantile(src.view(-1, self.n_dim), q, dim=0, interpolation="linear")
        self.median = nn.Parameter(quantile_points[1])
        self.delta = nn.Parameter(quantile_points[2] - quantile_points[0])

        # target distribution
        target_distribution = torch.distributions.normal.Normal(target_mean, target_std)
        target_quantile_points = target_distribution.icdf(q)
        self.target_median = nn.Parameter(target_quantile_points[1])
        self.target_delta = nn.Parameter(target_quantile_points[2] - target_quantile_points[0])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the input tensor to match the target distribution using quantile scaling .

        Args:
            tensor: The input tensor to transform.

        Returns:
            The transformed tensor, scaled to the target distribution.
        """
        out = (tensor - self.median) / self.delta
        out = out * self.target_delta + self.target_median

        return out

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the transformation applied by the forward method, mapping the tensor back to its original
        distribution.

        Args:
            tensor: The tensor to reverse the transformation on.

        Returns:
            The tensor with the quantile scaling transformation reversed according to the src distribution.
        """
        out = (tensor - self.target_median) / self.target_delta
        out = out * self.delta + self.median
        return out
