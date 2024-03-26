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


class CrossAttention(Attention):
    def __init__(self, d_model, num_heads, attention: Attention = None, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        if attention is None:
            self.attention = ScaledDotProductAttention()
        else:
            self.attention = attention
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def split_into_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # Linear projection and split into heads
        q = self.split_into_heads(self.Wq(q), batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_into_heads(self.Wk(k), batch_size)  # (batch_size, num_heads, seq_len_kv, depth)
        v = self.split_into_heads(self.Wv(v), batch_size)  # (batch_size, num_heads, seq_len_kv, depth)

        # apply attention
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer and dropout
        output = self.dropout(self.linear(attn_output))

        # Add & Norm
        output = self.layer_norm(output + q)

        return output, attn_weights
