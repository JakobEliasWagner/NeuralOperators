import warnings

import torch

from . import Transform


class ZNormalization(Transform):
    """Z-normalization transformation.

    This transformation takes the mean $\mu$ and the standard deviation $\sigma$ of the feature vector and scales it
    with $z(x)=\frac{x - \mu}{\sigma}$.

    Attributes:
        mean: mean value of the feature vector.
        std: standard deviation of the feature vector.
        epsilon: Value to prevent divide by zero.
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, epsilon: float = torch.finfo(torch.get_default_dtype()).tiny
    ):
        """

        Args:
            mean:
            std:
            epsilon:
        """
        super().__init__()
        self.mean = mean
        if torch.allclose(std, torch.zeros(std.shape)):
            warnings.warn(
                "Z-normalization with standard deviation 0! "
                "The feature vector does not have any discriminative power!",
                stacklevel=2,
            )
        self.std = std
        self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        $$\frac{x - \mu}{\sigma}$$

        Args:
            tensor:

        Returns:

        """
        return (tensor - self.mean) / (self.std + self.epsilon)

    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + self.epsilon) + self.mean
