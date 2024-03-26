from typing import (
    Tuple,
)

import torch
import torch.nn as nn

from .attention import (
    Attention,
)
from .scaled_dot_product_attention import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(Attention):
    def __init__(self, d_model, num_heads, attention: Attention = None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V transformations
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Scaled Dot-Product Attention
        if attention is None:
            self.attention = ScaledDotProductAttention()
        else:
            self.attention = attention

        # Final linear layer to concatenate heads
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # Perform linear operation and split into num_heads
        q = self.split_heads(self.linear_q(q), batch_size)
        k = self.split_heads(self.linear_k(k), batch_size)
        v = self.split_heads(self.linear_v(v), batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Make the mask suitable for multi-head broadcasting

        # Apply Scaled Dot-Product Attention
        attn, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and put through final linear layer
        attn = attn.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.out(attn), attn_weights
