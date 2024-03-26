from typing import (
    Tuple,
)

import torch
import torch.nn.functional as F

from .attention import (
    Attention,
)


class ScaledDotProductAttention(Attention):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

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
        d_k = q.size(-1)  # Dimension of the key
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        return torch.matmul(attention_weights, v), attention_weights
