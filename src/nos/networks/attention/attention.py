from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Tuple,
)

import torch
import torch.nn as nn


class Attention(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - Q: Queries tensor of shape (batch_size, num_queries, d_k)
        - K: Keys tensor of shape (batch_size, num_keys, d_k)
        - V: Values tensor of shape (batch_size, num_values, d_v) where typically num_keys == num_values
        - mask: Masking tensor of shape (batch_size, 1, num_keys) or (batch_size, num_queries, num_keys)

        Returns:
        - The attention weighted tensor and the attention weights.
        """
