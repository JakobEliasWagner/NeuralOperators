from abc import (
    ABC,
    abstractmethod,
)

import torch


class Indicator(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
              x: tensor of locations where the indicator is evaluated of shape (n_observations, n_dim).

        Returns:
            Tensor containing the indicator values of shape (n_observations).
        """
