import torch
import torch.nn as nn


class FunctionEncoder(nn.Module):
    def __init__(self, n_dim: int, n_head: int = 1, width_hidden: int = 32, dropout_p: float = 0):
        super().__init__()
        # attention
        self.attn = nn.MultiheadAttention(n_dim, n_head, bias=True)
        self.attn_norm = nn.LayerNorm(n_dim, eps=1e-5, bias=True)
        self.attn_dropout = nn.Dropout()

        # feed forward
        self.activation = nn.GELU()
        self.ff_norm = nn.LayerNorm(n_dim, eps=1e-5, bias=True)
        self.ff_lin_1 = nn.Linear(n_dim, width_hidden, bias=True)
        self.ff_lin_2 = nn.Linear(width_hidden, n_dim, bias=True)
        self.ff_dropout_1 = nn.Dropout(dropout_p)
        self.ff_dropout_2 = nn.Dropout(dropout_p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src

        # self attention block
        out = self.attn(out, out, out)[0]
        out = self.attn_dropout(out)
        out = out + src
        out = self.attn_norm(out)
        attn_out = out

        # feed forward block
        out = self.ff_lin_1(out)
        out = self.activation(out)
        out = self.ff_dropout_1(out)
        out = self.ff_lin_2(out)
        out = self.ff_dropout_2(out)
        out = out + attn_out
        out = self.ff_norm(out)

        return out
