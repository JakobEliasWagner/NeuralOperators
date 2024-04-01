from typing import (
    Callable,
)

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:


    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        attention: Callable = nn.functional.scaled_dot_product_attention,
        dropout_p: float = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention = attention
        self.dropout_p = dropout_p
        self.bias = bias

        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "hidden_dim must be divisible by n_heads"

        # projection networks
        self.query_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.key_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.value_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        # project values
        query = self.query_project(query)
        key = self.key_project(key)
        value = self.value_project(value)

        query = query.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key = key.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value = value.reshape(batch_size, self.num_heads, -1, self.head_dim)

        attn_out = self.attention(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=self.dropout_p)
        attn_out = attn_out.reshape(batch_size, -1, self.hidden_dim)
        return self.out_project(attn_out)
